import os
import numpy as np
import re  
import tempfile  
import subprocess  
import traceback  
import json
from hlseval.utils import get_cpp_type_str, import_module_functions


class NumpyEncoder(json.JSONEncoder):  
    """  
    Custom encoder for numpy objects to make them JSON serializable  
    """  
    def default(self, obj):  
        if isinstance(obj, np.ndarray):  
            return obj.tolist()  
        elif isinstance(obj, np.integer):  
            return int(obj)  
        elif isinstance(obj, np.floating):  
            return float(obj)  
        elif isinstance(obj, np.bool_):  
            return bool(obj)  
        elif isinstance(obj, (np.void, np.complex_)):  
            return str(obj)  
        return json.JSONEncoder.default(self, obj) 
   
class CsimTimeoutError(Exception):
    """Raised when Vitis HLS times out."""
    pass


def generate_h_file(kernel_code, model_dir, kernel_name):
    """
    Parses C++ kernel code to extract the function signature and generates a corresponding .h header file.
    This version is more robust to multi-line function declarations and whitespace.

    Args:
        kernel_code (str): The full C++ source code of the kernel.
        model_dir (str): The directory where the .h file should be saved.
        kernel_name (str): The name of the kernel function (e.g., "axpy_int32").
    """
    os.makedirs(model_dir, exist_ok=True)
    h_file_path = os.path.join(model_dir, f"{kernel_name}.h")

    lines = kernel_code.splitlines()

    preprocessor_lines = []
    func_signature_lines = []
    found_function_start = False
    brace_found = False
    func_signature = None

    function_name_pattern = re.compile(r'\b' + re.escape(kernel_name) + r'\s*\(')
    

    for i, line in enumerate(lines):
        stripped = line.strip()

        if not found_function_start:
            if stripped.startswith("#"):
                preprocessor_lines.append(line)
            else:
                # Accumulate potential signature lines until the kernel name is found
                # Or just start looking for kernel name if not a preprocessor
                # This ensures we capture the return type too if it's on a previous line
                pass # We'll build func_signature_lines once found_function_start is True

            # Check if the current line (or part of it) contains the function's start
            if function_name_pattern.search(line):
                found_function_start = True
                # Start collecting lines from this point for the signature
                func_signature_lines.append(line)
                # If the brace is on the same line as the function name start, handle it
                if '{' in line:
                    brace_idx = line.find('{')
                    full_signature_raw = "".join(func_signature_lines) # Join just what we have so far
                    brace_idx_in_full = full_signature_raw.find('{')
                    func_signature = full_signature_raw[:brace_idx_in_full].strip()
                    brace_found = True
                    break # We found everything we need

        elif found_function_start:
            # If function start was already detected, keep adding lines until the brace is found
            func_signature_lines.append(line)
            if '{' in line:
                brace_idx = line.find('{') # Find the brace in the current line
                # Join all collected lines for the complete signature
                full_signature_raw = "\n".join(func_signature_lines)
                brace_idx_in_full = full_signature_raw.find('{') # Find the first brace in the combined string
                if brace_idx_in_full == -1: # Should not happen if '{' in line was true
                    raise ValueError("Internal error: Opening brace '{' not found in combined signature lines.")
                func_signature = full_signature_raw[:brace_idx_in_full].strip()
                brace_found = True
                break # We found everything we need

    if not brace_found:
        raise ValueError(f"Opening brace '{{' for function '{kernel_name}' not found after its declaration in the provided code.")
    if 'func_signature' not in locals() or func_signature is None: # Check if func_signature was actually set
        raise ValueError(f"Function signature for '{kernel_name}' could not be extracted. Check kernel_code format.")



    if func_signature is None:
        raise ValueError(f"Function {kernel_name} not found in the provided code.")

    # Add semicolon to make it a declaration
    func_declaration = func_signature + ";"

    # Prepare include guard macro
    guard_macro = kernel_name.upper() + "_H"

    # Write everything to the header file
    with open(h_file_path, "w") as f:
        f.write(f"#ifndef {guard_macro}\n")
        f.write(f"#define {guard_macro}\n\n")

        # Write preprocessor lines if any
        if preprocessor_lines:
            f.write("\n".join(preprocessor_lines) + "\n\n")

        f.write(func_declaration + "\n\n")
        f.write(f"#endif // {guard_macro}\n")

    print(f"Header file generated at: {h_file_path}")

    pass

    
def create_testbench(kernel_name, test_data_json_path, model_dir):
    """
    Generates a C++ testbench file (`testbench.cpp`) that reads test data from a JSON file,
    executes the specified kernel, and compares its outputs to expected results.
    This function *does not* compile or run the generated C++ file.

    Args:
        kernel_name (str): Name of the kernel function (e.g., "axpy").
        test_data_json_path (str): Full path to the JSON file containing test cases.
        model_dir (str): Directory name within results/solutions/ containing the kernel files.

    Returns:
        tuple: (testbench_file_path, json_file_path, status_message)
               testbench_file_path (str): Path to the generated C++ testbench file.
               json_file_path (str): Path to the JSON test data file it relies on.
               status_message (str): "SUCCESS" if file generated, "FAILED" otherwise.
    """
    mismatches_to_report = 10
    project_root = os.path.dirname(os.path.dirname(__file__))
    cpp_kernel_file_abs_path = os.path.join(model_dir, f"{kernel_name}.cpp")

    if not os.path.exists(cpp_kernel_file_abs_path):
        print(f"C++ kernel file not found: {cpp_kernel_file_abs_path}")
        return None, test_data_json_path, f"FAILED: C++ kernel file not found: {cpp_kernel_file_abs_path}"

    # Create the path for the testbench file
    testbench_cpp_file_path = os.path.join(model_dir, f"{kernel_name}_tb.cpp")
    json_filename_only = os.path.basename(test_data_json_path)

    # --- Read types_metadata directly from the JSON file ---
    try:
        with open(test_data_json_path, 'r') as f:
            test_cases_from_json = json.load(f)
        if not test_cases_from_json:
            return None, json_filename_only, "FAILED: No test cases found in JSON.", None
        # Assuming only one test case is generated per file by sim_kernel
        all_var_metadata = test_cases_from_json[0].get("types_metadata", [])
        if not all_var_metadata:
            return None, json_filename_only, "FAILED: 'types_metadata' not found in JSON.", None
    except Exception as e:
        return None, json_filename_only, f"FAILED: Error reading or parsing JSON for types_metadata: {e}", None

    # --- Creating the h file for the kernel code ---
    # Read the kernel code  
    try:  
        with open(cpp_kernel_file_abs_path, 'r', encoding='utf-8') as f:  
            kernel_code = f.read()  
    except UnicodeDecodeError:  
        # Try with a different encoding if UTF-8 fails  
        with open(cpp_kernel_file_abs_path, 'r', encoding='latin-1') as f:  
            kernel_code = f.read()  
    # Generating H File
    print("--- Generating H File for CSIM ---")
    # h_file_path = os.path.join(project_root, "results", "solutions", model_dir)
    generate_h_file(kernel_code, model_dir, kernel_name)
    
    # --- C++ Testbench Code Generation ---
    testbench_code = []

    # Required Headers
    testbench_code.append("#include <iostream>")
    testbench_code.append("#include <fstream>")
    testbench_code.append("#include <vector>")
    testbench_code.append("#include <string>")
    testbench_code.append("#include <cmath>") # For std::abs
    testbench_code.append("#include <limits>") # For numeric_limits epsilon
    testbench_code.append("#include \"nlohmann/json.hpp\"") # Assuming nlohmann/json.hpp is available in include path
    testbench_code.append("#include \"utils.hpp\"")    # Floating-point comparison helper function

    # Include the actual kernel file using its absolute path (or a relative path if copied)
    # The generated testbench.cpp needs to know where kernel_name.cpp is.
    # For clarity, we'll suggest copying the kernel.cpp to temp_dir if using relative include.
    # For now, we'll assume the external script will either copy it or use -I.
    # Using a direct include of the name, assuming the compiler knows its location.
    testbench_code.append(f"#include \"{kernel_name}.h\"\n\n")


    testbench_code.append("using json = nlohmann::json;")
    testbench_code.append("using namespace std;\n\n")

    testbench_code.append("int main() {")

    # Open and parse JSON test data
    # NOTE: The JSON file path needs to be accessible from where the C++ executable runs.
    # It's safer to pass this path as a command-line argument, but for now,
    # let's assume it's hardcoded and the external script handles copying/location.
    # A better approach would be: `std::string json_path = argv[1];`
    # For simplicity, we hardcode it for now, and the external script must copy it.
    # We will pass the *relative* path if the external script copies the JSON to temp_dir.
    json_filename_only = os.path.basename(test_data_json_path)
    testbench_code.append(f"    std::ifstream ifs(\"{test_data_json_path}\");")
    testbench_code.append("    if (!ifs.is_open()) {")
    testbench_code.append(f"        std::cerr << \"Error: Could not open test data file {json_filename_only}. Make sure it's in the executable's directory.\" << std::endl;")
    testbench_code.append("        return 1;")
    testbench_code.append("    }")
    testbench_code.append("    json test_cases_json;")
    testbench_code.append("    try {")
    testbench_code.append("        ifs >> test_cases_json;")
    testbench_code.append("    } catch (const json::parse_error& e) {")
    testbench_code.append("        std::cerr << \"JSON parse error: \" << e.what() << std::endl;")
    testbench_code.append("        return 1;")
    testbench_code.append("    }")

    testbench_code.append("\n    int total_tests = 0;")
    testbench_code.append("    int failed_tests = 0;")
    testbench_code.append("    int flat_idx = 0;")
    testbench_code.append("    float percent_error = 0.0;")


    # Loop through each test case
    testbench_code.append("\n    for (const auto& test_case : test_cases_json) {")
    testbench_code.append("        total_tests++;")
    testbench_code.append("        std::string test_id = test_case.at(\"test_id\").get<std::string>();")
    testbench_code.append("        std::cout << \"\\n--- Running Test Case: \" << test_id << \" ---\" << std::endl;")

    # Declarations, Allocations, and Initialization of Inputs/Outputs for C++
    kernel_call_args = []
    cleanup_deallocations = [] # To store delete[] statements for current test case


    for var_info in all_var_metadata:
        var_name = var_info["name"]
        array_dimension = var_info["array_dimension"]
        in_out_status = var_info["in_out"]
        dtype_str = var_info["dtype"] # e.g., "int32", "float64", "int", "float"
        cpp_base_type = get_cpp_type_str(dtype_str)
        shape = var_info.get('shape', []) # Shape from metadata (e.g., [10, 20])
        total_elements = var_info.get('total_elements', 1) # Total elements for flattening


        # Handle scalar (dimension 0)
        if array_dimension == 0:
            if in_out_status in ["in", "inout"]:
                testbench_code.append(f"        {cpp_base_type} {var_name}_val = test_case.at(\"inputs\").at(\"{var_name}\").get<{cpp_base_type}>();")
                kernel_call_args.append(var_name + "_val")
            if in_out_status in ["out", "inout"]:
                # For outputs, declare a variable and pass its address
                testbench_code.append(f"        {cpp_base_type} {var_name}_out_actual = 0; // Initialize for output clarity")
                kernel_call_args.append(f"&{var_name}_out_actual") # Pass by address for output
                # No dynamic allocation for scalar, so no cleanup here
        # Handle arrays (1D) use dynamic allocation and flatten
        elif array_dimension == 1:
            # Declare and allocate flattened array
            testbench_code.append(f"        {cpp_base_type}* {var_name}_arr = new {cpp_base_type}[{total_elements}];")
            cleanup_deallocations.append(f"        delete[] {var_name}_arr;")

            # Initialize from JSON for inputs
            if in_out_status in ["in", "inout"]:
                testbench_code.append(f"\n        // Initialize {var_name}_arr from JSON input")
                testbench_code.append(f"        auto {var_name}_json_data = test_case.at(\"inputs\").at(\"{var_name}\");")
                testbench_code.append(f"        if (!{var_name}_json_data.is_array() || {var_name}_json_data.size() < {total_elements}) {{")
                testbench_code.append(f"            std::cerr << \"Error: {var_name} is not an array or too small.\" << std::endl;")
                testbench_code.append(f"            return 1; // Or handle appropriately")
                testbench_code.append("        }")
                testbench_code.append(f"        for (int i = 0; i < {total_elements}; ++i) {{")
                testbench_code.append(f"            {var_name}_arr[i] = {var_name}_json_data.at(i).get<{cpp_base_type}>();")
                testbench_code.append("        }")
            elif in_out_status == "out":
                # Initialize output buffers to 0 for clarity
                testbench_code.append(f"\n        // Initialize output array {var_name}_arr to zeros")
                testbench_code.append(f"        for (int i = 0; i < {total_elements}; ++i) {{")
                testbench_code.append(f"            {var_name}_arr[i] = 0; // Or appropriate default value")
                testbench_code.append("        }")

            # Add to kernel arguments (passing pointer to flattened array)
            kernel_call_args.append(var_name + "_arr")
        
        

        # Handle 2D, 3D, 4D arrays (using fixed-size declarations for the testbench)
        else:
            # Construct the C++ array declaration string: e.g., "my_array[10][20]"
            cpp_array_dims = '[' + ']['.join(map(str, shape)) + ']'
            
            testbench_code.append(f"\n        // Declare {array_dimension}D array {var_name}")
            testbench_code.append(f"        {cpp_base_type} {var_name}_arr{cpp_array_dims};")

            # Initialize from JSON input
            if in_out_status in ["in", "inout"]:
                testbench_code.append(f"\n        // Initialize {var_name}_arr from JSON input (NESTED LOOPS)")
                testbench_code.append(f"        auto {var_name}_json_data = test_case.at(\"inputs\").at(\"{var_name}\");")
                
                testbench_code.append(f"        if (!{var_name}_json_data.is_array() || {var_name}_json_data.size() < {shape[0]}) {{")
                testbench_code.append(f"            std::cerr << \"Error: {var_name} is not a {array_dimension}D array (wrong number of outer elements).\" << std::endl;")
                testbench_code.append(f"            return 1; // Or handle appropriately")
                testbench_code.append("        }")          
                
                json_access_str = f"{var_name}_json_data"      
                # Generate nested loops for initialization
                indent = "            "
                loop_vars = [f"i{d}" for d in range(array_dimension)]
                for d in range(array_dimension):
                    testbench_code.append(f"{indent}for (int {loop_vars[d]} = 0; {loop_vars[d]} < {shape[d]}; ++{loop_vars[d]}) {{")
                    indent += "    "
                    
                    if d < array_dimension - 1: # If not the innermost loop
                        json_access_str += f".at({loop_vars[d]})"
                        testbench_code.append(f"{indent}    auto current_level_json_{loop_vars[d]} = {json_access_str};")
                        testbench_code.append(f"{indent}    if (!current_level_json_{loop_vars[d]}.is_array() || current_level_json_{loop_vars[d]}.size() < {shape[d+1]}) {{")
                        testbench_code.append(f"{indent}        std::cerr << \"Error: {var_name} at level {d} ({loop_vars[d]}) is not an array or too small.\" << std::endl;")
                        testbench_code.append(f"{indent}        return 1; // Or handle appropriately")
                        testbench_code.append(f"{indent}    }}")
                        json_access_str = f"current_level_json_{loop_vars[d]}" # Update for next level
                                    
                # Assignment line
                cpp_access_str = ''.join([f"[{lv}]" for lv in loop_vars])
                testbench_code.append(f"{indent}{var_name}_arr{cpp_access_str} = {json_access_str}.at({loop_vars[-1]}).get<{cpp_base_type}>();") # Use last loop_var for access
                
                # Close loops
                for d in range(array_dimension - 1, -1, -1):
                    indent = indent[:-4]
                    testbench_code.append(f"{indent}}}")
            
            elif in_out_status == "out":
                testbench_code.append(f"\n        // Initialize output {array_dimension}D array {var_name}_arr to zeros")
                indent = "            "
                loop_vars = [f"i{d}" for d in range(array_dimension)]
                for d in range(array_dimension):
                    testbench_code.append(f"{indent}for (int {loop_vars[d]} = 0; {loop_vars[d]} < {shape[d]}; ++{loop_vars[d]}) {{")
                    indent += "    "
                cpp_access_str = ''.join([f"[{lv}]" for lv in loop_vars])
                testbench_code.append(f"{indent}{var_name}_arr{cpp_access_str} = 0; // Or appropriate default value")
                for d in range(array_dimension - 1, -1, -1):
                    indent = indent[:-4]
                    testbench_code.append(f"{indent}}}")
            
            kernel_call_args.append(var_name + "_arr")

            

    # Call the kernel function
    testbench_code.append(f"\n        // Call the kernel function")
    testbench_code.append(f"        {kernel_name}({', '.join(kernel_call_args)});")

    # --- Comparison Logic ---
    testbench_code.append("\n        bool test_case_passed = true;")
    testbench_code.append("        std::cout << \"Comparing results...\" << std::endl;")


    # Compare outputs based on all_var_metadata
    for var_info in all_var_metadata:
        var_name = var_info["name"]
        array_dimension = var_info["array_dimension"]
        in_out_status = var_info["in_out"]
        dtype_str = var_info["dtype"]
        cpp_base_type = get_cpp_type_str(dtype_str)
        shape = var_info.get('shape', []) # Shape from metadata (e.g., [10, 20])

        if in_out_status in ["out", "inout"]:
            # Get expected values from JSON
            testbench_code.append(f"\n        uint32_t num_mismatches = 0;")
            
            testbench_code.append(f"\n        // Get expected values for {var_name}")
            testbench_code.append(f"        auto {var_name}_expected_json = test_case.at(\"expected_outputs\").at(\"{var_name}\");")
            
            # Check for the outer-most dimension
            if array_dimension > 0: # Only check for arrays
                testbench_code.append(f"        if (!{var_name}_expected_json.is_array() || {var_name}_expected_json.size() < {shape[0]}) {{")
                testbench_code.append(f"            std::cerr << \"Error: expected_outputs.{var_name} is not a {array_dimension}D array (wrong number of outer elements).\" << std::endl;")
                testbench_code.append(f"            return 1; // Or handle appropriately")
                testbench_code.append("        }")
                
            if array_dimension == 0: # Scalar comparison
                testbench_code.append(f"        {cpp_base_type} {var_name}_expected_val = {var_name}_expected_json.get<{cpp_base_type}>();")
                testbench_code.append(f"        if (!compare_numeric({var_name}_out_actual, {var_name}_expected_val)) {{")
                # FIXME: Commentted out these lines that cause context window to explode with failed test cases. Need to know if I should replace them with something else.
                testbench_code.append(f"            if (num_mismatches < {mismatches_to_report}){{")
                testbench_code.append(f"                std::cout << \"  FAIL: Scalar '{var_name}' - Expected: \" << {var_name}_expected_val << \", Actual: \" << {var_name}_out_actual << std::endl;")
                testbench_code.append("             }")
                testbench_code.append("             num_mismatches++;")
                testbench_code.append("             test_case_passed = false;")
                testbench_code.append("         }")
                testbench_code.append(f"        else {{ std::cout << \"  PASS: Scalar '{var_name}' matches.\" << std::endl; }}")


            elif array_dimension == 1: # Array comparison (1D)
                testbench_code.append(f"\n        // Compare array {var_name}")
                testbench_code.append(f"        for (int i = 0; i < {total_elements}; ++i) {{")
                testbench_code.append(f"            {cpp_base_type} expected_val = {var_name}_expected_json.at(i).get<{cpp_base_type}>();")
                testbench_code.append(f"            if (!compare_numeric({var_name}_arr[i], expected_val)) {{")                
                testbench_code.append(f"                if (num_mismatches < {mismatches_to_report}){{")
                testbench_code.append(f"                    std::cout << \"  FAIL: Array '{var_name}' at flattened index [\" << i << \"] - Expected: \" << expected_val << \", Actual: \" << {var_name}_arr[i] << std::endl;")
                testbench_code.append("                 }")
                testbench_code.append("                 num_mismatches++;")
                testbench_code.append("                 test_case_passed = false;")
                testbench_code.append("                 // break; // Uncomment to stop on first mismatch for this array")
                testbench_code.append("             }")
                testbench_code.append("         }")
                testbench_code.append(f"        if (test_case_passed) std::cout << \"  PASS: Array '{var_name}' matches.\" << std::endl;")
                

            else: # 2D, 3D, 4D array comparison
                testbench_code.append(f"\n        // Compare {array_dimension}D array {var_name}")

                indent = "        "
                json_access_str = f"{var_name}_expected_json"
                loop_vars = [f"j{d}" for d in range(array_dimension)] # Use j_ to avoid conflicts with i_ in other loops
                
                for d in range(array_dimension):
                    indent += "    "
                    testbench_code.append(f"{indent}for (int {loop_vars[d]} = 0; {loop_vars[d]} < {shape[d]}; ++{loop_vars[d]}) {{")
                    
                    if d < array_dimension - 1: # If not the innermost loop
                        json_access_str += f".at({loop_vars[d]})"
                        testbench_code.append(f"{indent}    auto current_level_expected_json_{loop_vars[d]} = {json_access_str};")
                        testbench_code.append(f"{indent}    if (!current_level_expected_json_{loop_vars[d]}.is_array() || current_level_expected_json_{loop_vars[d]}.size() < {shape[d+1]}) {{")
                        testbench_code.append(f"{indent}        std::cerr << \"Error: expected_outputs.{var_name} at level {d} ({loop_vars[d]}) is not an array or too small.\" << std::endl;")
                        testbench_code.append(f"{indent}        return 1; // Or handle appropriately")
                        testbench_code.append(f"{indent}    }}")
                        json_access_str = f"current_level_expected_json_{loop_vars[d]}" # Update for next level

                cpp_access_str = ''.join([f"[{lv}]" for lv in loop_vars])
                cout_loop_vars = "<<".join([f" \"[\" << {lv} <<  \"]\"" for lv in loop_vars])  # For output formatting
                testbench_code.append(f"{indent}{cpp_base_type} expected_val = {json_access_str}.at({loop_vars[-1]}).get<{cpp_base_type}>();")
                testbench_code.append(f"{indent}if (!compare_numeric({var_name}_arr{cpp_access_str}, expected_val)) {{")
                testbench_code.append(f"{indent}    if (num_mismatches < {mismatches_to_report}){{")
                testbench_code.append(f"{indent}        std::cout << \"  FAIL: {array_dimension}D Array '{var_name}' at index\" << {cout_loop_vars} << \"- Expected: \" << expected_val << \", Actual: \" << {var_name}_arr{cpp_access_str} << std::endl;")
                testbench_code.append(f"{indent}    }}")
                testbench_code.append(f"{indent}    test_case_passed = false;")
                testbench_code.append(f"{indent}    num_mismatches++;")

                testbench_code.append(f"{indent}    // goto end_comparison_for_{var_name}; // Consider using for early exit on mismatch")
                testbench_code.append(f"{indent}}}")
                
                for d in range(array_dimension - 1, -1, -1):
                    indent = indent[:-4]
                    testbench_code.append(f"{indent}}}")
                # testbench_code.append(f"        // end_comparison_for_{var_name}:;") # Label for goto
                testbench_code.append(f"        if (test_case_passed) std::cout << \"  PASS: {array_dimension}D Array '{var_name}' matches.\" << std::endl;")

                

    testbench_code.append("\n        if (test_case_passed) {")
    testbench_code.append("            std::cout << \"Test Case PASSED: \" << test_id << std::endl;")
    testbench_code.append("        } else {")
    testbench_code.append("            std::cout << \"Test Case FAILED: \" << test_id << std::endl;")
    testbench_code.append("            failed_tests++;")
    testbench_code.append("        }")


    # Cleanup dynamic memory for current test case
    for cleanup_line in cleanup_deallocations:
        testbench_code.append(cleanup_line)

    #Calculate the Percent Error
    testbench_code.append("        if (num_mismatches > 0) {")
    testbench_code.append(f"            percent_error = ((float)num_mismatches / {total_elements}) * 100.0;")
    testbench_code.append("        } else {")
    testbench_code.append("            percent_error = 0.0;")
    testbench_code.append("        }")

    testbench_code.append("    }") # End of for-loop for test_cases_json
    




    testbench_code.append("\n    std::cout << \"\\n--- Test Summary ---\" << std::endl;")
    testbench_code.append("    std::cout << \"Compilation Status: success\" << std::endl;")
    testbench_code.append("    std::cout << \"Total Tests: \" << total_tests << std::endl;")
    testbench_code.append("    std::cout << \"Failed Tests: \" << failed_tests << std::endl;")
    testbench_code.append("    std::cout << \"Percent Error: \" << percent_error << \"%\" << std::endl;")
    testbench_code.append("    if (failed_tests == 0) {")
    testbench_code.append("        std::cout << \"ALL TESTS PASSED!\" << std::endl;")
    testbench_code.append("        return 0;")
    testbench_code.append("    } else {")
    testbench_code.append("        std::cout << \"SOME TESTS FAILED!\" << std::endl;")
    testbench_code.append("        return 1;")
    testbench_code.append("    }")
    testbench_code.append("}") # End of main


    # --- Write the generated testbench.cpp file ---
    try:
        with open(testbench_cpp_file_path, "w") as f:
            f.write("\n".join(testbench_code))
        print(f"Generated C++ testbench source file: {testbench_cpp_file_path}")
        return testbench_cpp_file_path, test_data_json_path, "SUCCESS"
    except FileExistsError as e:
        print(f"C++ Testbench {testbench_cpp_file_path} already exists.")
        return testbench_cpp_file_path, test_data_json_path, f"SUCCESS: Testbench already exists: {e}"
    except Exception as e:
        print(f"Error writing C++ testbench file: {e}")
        return None, test_data_json_path, f"FAILED: Error writing C++ testbench file: {e}"


def write_json_test_data(filename, test_cases_list):
    """
    Writes a list of test cases (inputs and expected outputs) to a JSON file.
    """
    try:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"Created directory: {output_dir}")

        with open(filename, 'w') as f:
            json.dump(test_cases_list, f, indent=4, cls=NumpyEncoder)
        print(f"Successfully wrote test data to {filename}")
    except Exception as e:
        print(f"Error writing test data to {filename}: {e}")
        raise


def create_reference_data(kernel_name=None, model_dir="./results/solutions/gpt-4o/attempts_1", seed=0):
    """
    Writes a list of test cases (inputs and expected outputs) to a JSON file.

    Each test case should be a dictionary with at least 'test_id', 'inputs',
    and 'expected_outputs' keys.

    Args:
        kernel_name (str): The name of the kernel being tested
                        Example: "mat_mult_int16"
        model_dir (str): The directory to which we will be creating the json data file. 
                        Example: "gpt-4o/attempts_1"
        seed (int): The seed number for the random number generation. 
    Raises:
        IOError: If there's an issue opening or writing to the file.
        TypeError: If the data cannot be serialized to JSON.
    """
    project_root = (os.path.dirname(os.path.dirname(__file__)))
    kernel_dir = os.path.join(project_root, "dataset", "kernels")

    if kernel_name is None:
        kernel_names = [d for d in os.listdir(kernel_dir)
                      if os.path.isdir(os.path.join(kernel_dir, d))]
    else:
        kernel_names = [kernel_name]
        
    all_results = {} # Changed to all_results to avoid confusion with single kernel results

    for name in kernel_names:
        print(f"\n--- Processing kernel for JSON reference data: {name} ---")
        
        functions = import_module_functions(name, ["generate_inputs", "behavioral"])
        
        
        if not functions or "generate_inputs" not in functions:
            print(f"Could not import generate_inputs for {name}, skipping...")
            all_results[name] = {"status": "error", "message": "Could not import generate_inputs"}
            continue

        generate_inputs_func = functions["generate_inputs"]
        generation_result = generate_inputs_func(np.int64(seed))
        
        if isinstance(generation_result, dict) and "inputs" in generation_result:
            inputs_list = generation_result["inputs"]
            input_types = generation_result.get("types")
        else:
            inputs_list = generation_result # Old format
            input_types = None # No type info

        print(f"Generated raw inputs for {name}")

        # Separate 'in' arguments for behavioral model call
        input_arg_values_for_behavioral = []
        # Keep track of names for inputs to JSON
        json_inputs_dict = {}
        # Keep track of variable names for outputs to JSON
        json_expected_outputs_dict = {}
        # List of output variable names in order for the C++ testbench to know
        output_var_names_in_order = []

        # Populate json_inputs_dict and prepare for behavioral call
        for i, type_info in enumerate(input_types):
            var_name = type_info["name"]
            current_value = inputs_list[i] # Get the actual value from the inputs_list

            if type_info["in_out"] in ["in", "inout"]:
                # These are inputs to the kernel, so they go into the JSON's "inputs"
                json_inputs_dict[var_name] = current_value
                # Also collect for behavioral function call (copy if mutable)
                input_arg_values_for_behavioral.append(np.copy(current_value) if isinstance(current_value, np.ndarray) else current_value)

            if type_info["in_out"] in ["out", "inout"]:
                # These are outputs from the kernel. We need their names to map behavioral results.
                json_inputs_dict[var_name] = current_value
                # All inputs and outputs are inputs to the c++ kernel. 
                output_var_names_in_order.append(var_name)
                # json_expected_outputs_dict[var_name] will be filled after behavioral run
            # For 'in' only types, they don't get added to json_expected_outputs_dict
            # For 'out' only types, they are not added to input_arg_values_for_behavioral
        
        expected_result_from_behavioral = None
        if "behavioral" in functions:
            behavioral_func = functions["behavioral"]
            try:
                # Call behavioral with only the 'in' and 'inout' args, which are collected in order
                behavioral_result_raw = behavioral_func(*input_arg_values_for_behavioral)
                print(f"Behavioral model executed successfully for {name}")

                # Map behavioral_result_raw to json_expected_outputs_dict
                if isinstance(behavioral_result_raw, tuple):
                    if len(behavioral_result_raw) != len(output_var_names_in_order):
                        raise ValueError(
                            f"Behavioral model returned {len(behavioral_result_raw)} outputs, "
                            f"but expected {len(output_var_names_in_order)} based on input_types for {name}. "
                            f"Expected output names: {output_var_names_in_order}"
                        )
                    for i, output_name in enumerate(output_var_names_in_order):
                        json_expected_outputs_dict[output_name] = behavioral_result_raw[i]
                else: # Single output case
                    if len(output_var_names_in_order) != 1:
                        # This means behavioral returned a single value, but multiple outputs are expected
                        raise ValueError(
                            f"Behavioral model returned a single output, but expected {len(output_var_names_in_order)} "
                            f"outputs based on input_types for {name}. Expected output names: {output_var_names_in_order}"
                        )
                    json_expected_outputs_dict[output_var_names_in_order[0]] = behavioral_result_raw

            except Exception as e:
                print(f"Error running behavioral function for {name}: {e}")
                all_results[name] = {"status": "error", "message": f"Behavioral error: {e}"}
                continue
        else:
            print(f"No behavioral model found for {name}, skipping JSON generation.")
            all_results[name] = {"status": "error", "message": "No behavioral model"}
            continue

        # Construct the single test case dictionary
        current_test_case = {
            "test_id": f"{name}_test_{seed}",
            "inputs": json_inputs_dict,
            "expected_outputs": json_expected_outputs_dict,
            "types_metadata": input_types # Include full types info for C++ to interpret easily
        }

        # The JSON file path.
        json_output_filename = os.path.abspath(os.path.join(model_dir, f"{name}_test_data.json"))

        # Write this single test case to a JSON file
        try:
            # write_json_test_data expects a list of test cases, even if it's just one
            write_json_test_data(json_output_filename, [current_test_case])
            print(f"Generated JSON test data for {name} at: {json_output_filename}")
        except Exception as e:
            print(f"Failed to write JSON test data for {name}: {e}")
            all_results[name] = {"status": "error", "message": f"JSON write error: {e}"}
            continue


        # --- NOW CALL THE NEW sim_cpp_with_inputs (Testbench File Generator) ---
        print(f"\n--- Generating C++ testbench source file for {name} ---")
        testbench_cpp_path, json_filename_basename, generation_status = create_testbench(
            kernel_name=name,
            test_data_json_path=json_output_filename, # This is the full path to the source JSON
            model_dir=model_dir
        )

        if generation_status == "SUCCESS" and testbench_cpp_path:
            all_results[name] = {
                "status": "success_testbench_generated",
                "testbench_cpp_path": testbench_cpp_path,
                "message": "C++ testbench source file generated successfully.",
            }
        elif "SUCCESS" in generation_status:
            all_results[name] = {
                "status": "success_testbench_generated_with_warnings",
                "testbench_cpp_path": testbench_cpp_path,
                "message": generation_status
                
            }
        else:
            all_results[name] = {
                "status": "error_testbench_generation",
                "json_path": os.path.basename(json_output_filename), # In case temp_dir_for_files is None
                "message": generation_status
            }

    return all_results

def run_csim(kernel_name=None, model_dir="./results/solutions/gpt-4o/axpy_int32/attempts_1"):
    print("--- Preparing CSIM ---")
    result = create_reference_data(kernel_name, model_dir, seed=45)
    
    project_dir = model_dir
    kernel_code_path = os.path.join(project_dir, f"{kernel_name}.cpp")
    kernel_testbench_path = os.path.join(project_dir, f"{kernel_name}_tb.cpp")
    
    with open("vitis_config_csim.cfg", "w") as file:
        file.write(
            f"""
        part=xazu1eg-sbva484-1-i

        [hls]
        flow_target=vitis
        package.output.format=xo
        syn.top={kernel_name}
        syn.file={kernel_code_path}
        tb.file={kernel_testbench_path}
        tb.cflags=-fexceptions -I./include
        csim.code_analyzer=1
        csim.clean=true
        """ )
    
    sim_cmd = ["vitis-run", "--mode", "hls", "--csim", "--config", "vitis_config_csim.cfg", "--work_dir", project_dir]
    # Run the vitis_hls and capture the output
    result_success = "success" in result[kernel_name]["status"]
    if result_success:
        try:
            print("--- Running CSIM ---")
            sim_result = subprocess.run(
                sim_cmd,
                capture_output=True,
                check=False,
                timeout=300,  # Timeout after 2 minutes
            )
            sim_stdout = sim_result.stdout.decode('utf-8', errors='replace')  
            sim_stderr = sim_result.stderr.decode('utf-8', errors='replace') 
            return sim_result

        except subprocess.TimeoutExpired as e:
            print(f"CSIM timed out: {e}")
            # IMPORTANT: Access stdout and stderr from the exception object 'e'
            # Decode them just like you would from the result object
            sim_stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
            sim_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
            raise CsimTimeoutError(
                f"CSIM timed out after {e.timeout} seconds. Output: {sim_stdout}, Error: {sim_stderr}"
            )


        except subprocess.CalledProcessError as e: # Catch this specifically for non-zero exit codes
            print(f"CSIM failed with exit code {e.returncode}")
            sim_stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
            sim_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
            # print(sim_stdout)
            return e

        except Exception as e:
            print(f"An unexpected error occurred during CSIM: {e}")
            return e
                
# --- Main execution block ---
if __name__ == "__main__":
    # Ensure this path exists or is creatable
    target_model_dir = "./results/solutions/gpt-4o/mat_mult_int16/attempts_1"

    run_csim("mat_mult_int16", target_model_dir)
    # After running this, inspect the generated JSON file
    # Example: cat results/solutions/gpt-4o/axpy_int32/attempts_6/axpy_int32_test_data.json
