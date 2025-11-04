import logging
import os
import numpy as np
import re  
import subprocess  
import json
from datetime import datetime as dt
from hlseval.utils import get_ap_cpp_type_str, import_module_members, get_std_cpp_type_str, generate_inputs

logger = logging.getLogger("CSIM")

class CsimTimeoutError(Exception):
    """Raised when Vitis HLS times out."""
    pass

class NumpyEncoder(json.JSONEncoder):  
    """  
    Custom encoder for numpy objects to make them JSON serializable  
    """  
    def default(self, obj):  
        if isinstance(obj, np.ndarray):  
            return obj.tolist()  
        elif isinstance(obj, np.integer):  
            return int(obj)  
        if hasattr(obj, 'dtype') and str(obj.dtype) == 'bfloat16':
            return float(obj) # Convert bfloat16 to a standard float
        elif isinstance(obj, np.floating):  
            return float(obj)  
        elif isinstance(obj, np.bool_):  
            return bool(obj)  
        elif isinstance(obj, (np.void, np.complex128)):  
            return str(obj)  
        return json.JSONEncoder.default(self, obj) 

class CsynthTimeoutError(Exception):
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
            # Check if the current line (or part of it) contains the function's start
            if function_name_pattern.search(line):
                found_function_start = True
                # Start collecting lines from this point for the signature
                func_signature_lines.append(line)
                # If the brace is on the same line as the function name start, handle it
                if '{' in line:
                    full_signature_raw = "".join(func_signature_lines) # Join just what we have so far
                    brace_idx_in_full = full_signature_raw.find('{')
                    func_signature = full_signature_raw[:brace_idx_in_full].strip()
                    brace_found = True
                    break # We found everything we need

        elif found_function_start:
            # If function start was already detected, keep adding lines until the brace is found
            func_signature_lines.append(line)
            if '{' in line:
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

    func_declaration = func_signature + ";"
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

    logger.info(f"Header file generated at: {h_file_path}")

def create_testbench(kernel_name, test_data_json_path, model_dir, hls_flow="vitis"):
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
    cpp_kernel_file_abs_path = os.path.join(model_dir, f"{kernel_name}.cpp")

    if not os.path.exists(cpp_kernel_file_abs_path):
        logger.error(f"C++ kernel file not found: {cpp_kernel_file_abs_path}")
        return None, test_data_json_path, f"FAILED: C++ kernel file not found: {cpp_kernel_file_abs_path}", None

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
    logger.info("CSIM: Generating H File for CSIM")
    try:
        generate_h_file(kernel_code, model_dir, kernel_name)
    except Exception as e:
        logger.error(f"Error writing C++ testbench file: {e}")
        return None, test_data_json_path, f"FAILED: Error writing C++ testbench file: {e}", None
    
    # --- C++ Testbench Code Generation ---
    testbench_code = []

    # Required Headers
    testbench_code.append("#include <iostream>")
    testbench_code.append("#include <fstream>")
    testbench_code.append("#include <vector>")
    testbench_code.append("#include <string>")
    testbench_code.append("#include <cmath>") # For std::abs
    if hls_flow == "vitis":
        testbench_code.append("#include <ap_int.h>") # For ap_int
        testbench_code.append("#include <ap_float.h>") # For ap_float
    testbench_code.append("#include <limits>") # For numeric_limits epsilon
    testbench_code.append("#include \"nlohmann/json.hpp\"") # Assuming nlohmann/json.hpp is available in include path
    testbench_code.append("#include \"utils.hpp\"")    # Floating-point comparison helper function
    testbench_code.append("#include \"nlohmann/apint_serializer.hpp\"")    # ap_int serializer
    # Include the actual kernel file 
    testbench_code.append(f"#include \"{kernel_name}.h\"\n\n")
    testbench_code.append("using json = nlohmann::json;")
    testbench_code.append("using namespace std;\n\n")
    testbench_code.append("int main() {")
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
    testbench_code.append("    uint32_t num_mismatches = 0;")
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
        in_out_status = var_info["iotype"]
        dtype_str = var_info["dtype"] # e.g., "int32", "float64", "int", "float"
        cpp_base_type = get_ap_cpp_type_str(dtype_str) if hls_flow == "vitis" else get_std_cpp_type_str(dtype_str)
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
        elif array_dimension >= 1:
            # Declare and allocate flattened array
            testbench_code.append(f"        {cpp_base_type}* {var_name}_arr = new {cpp_base_type}[{total_elements}];")
            cleanup_deallocations.append(f"        delete[] {var_name}_arr;")
            if in_out_status in ["in", "inout"]:
                testbench_code.append(f"\n        // Initialize {var_name}_arr from JSON input")
                testbench_code.append(f"        auto {var_name}_json_data = test_case.at(\"inputs\").at(\"{var_name}\");")
                testbench_code.append(f"        uint32_t flat_idx_for_{var_name} = 0;")
                testbench_code.append(f"        for (const auto& val : {var_name}_json_data) {{")
                testbench_code.append(f"            if (flat_idx_for_{var_name} >= {total_elements}) {{")
                testbench_code.append(f"                std::cerr << \"Error: {var_name} JSON data has more elements than expected ({total_elements}).\" << std::endl;")
                testbench_code.append(f"                return 1;")
                testbench_code.append(f"            }}")
                testbench_code.append(f"            {var_name}_arr[flat_idx_for_{var_name}++] = val.get<{cpp_base_type}>();")
                testbench_code.append("        }")
                testbench_code.append(f"        if (flat_idx_for_{var_name} != {total_elements}) {{")
                testbench_code.append(f"            std::cerr << \"Error: {var_name} JSON data has fewer elements than expected ({total_elements}).\" << std::endl;")
                testbench_code.append(f"            return 1;")
                testbench_code.append(f"        }}")      
            elif in_out_status == "out":
                testbench_code.append(f"\n        // Initialize output array {var_name}_arr to zeros")
                testbench_code.append(f"        for (int i = 0; i < {total_elements}; ++i) {{")
                testbench_code.append(f"            {var_name}_arr[i] = 0; // Or appropriate default value")
                testbench_code.append("        }")
            kernel_call_args.append(var_name + "_arr")
    testbench_code.append(f"\n        // Call the kernel function")
    testbench_code.append(f"        {kernel_name}({', '.join(kernel_call_args)});")
    testbench_code.append("\n        bool test_case_passed = true;")
    testbench_code.append("        std::cout << \"Comparing results...\" << std::endl;")

    # Compare outputs based on all_var_metadata
    for var_info in all_var_metadata:
        var_name = var_info["name"]
        array_dimension = var_info["array_dimension"]
        in_out_status = var_info["iotype"]
        dtype_str = var_info["dtype"]
        cpp_base_type = get_ap_cpp_type_str(dtype_str) if hls_flow == "vitis" else get_std_cpp_type_str(dtype_str)
        shape = var_info.get('shape', []) # Shape from metadata (e.g., [10, 20])
        total_elements = var_info.get('total_elements', 1) # Total elements for flattening
        
        if in_out_status in ["out", "inout"]:
            # Get expected values from JSON
            testbench_code.append(f"\n        num_mismatches = 0;")
            testbench_code.append(f"\n        // Get expected values for {var_name}")
            testbench_code.append(f"        auto {var_name}_expected_json = test_case.at(\"expected_outputs\").at(\"{var_name}\");")

            if array_dimension == 0: # Scalar comparison
                testbench_code.append(f"        {cpp_base_type} {var_name}_expected_val = {var_name}_expected_json.get<{cpp_base_type}>();")
                testbench_code.append(f"        if (!compare_numeric({var_name}_out_actual, {var_name}_expected_val)) {{")
                testbench_code.append(f"            if (num_mismatches < {mismatches_to_report}){{")
                testbench_code.append(f"                std::cout << \"  FAIL: Scalar '{var_name}' - Expected: \" << {var_name}_expected_val << \", Actual: \" << {var_name}_out_actual << std::endl;")
                testbench_code.append("             }")
                testbench_code.append("             num_mismatches++;")
                testbench_code.append("             test_case_passed = false;")
                testbench_code.append("         }")
                testbench_code.append(f"        else {{ std::cout << \"  PASS: Scalar '{var_name}' matches.\" << std::endl; }}")
            elif array_dimension >= 1: # Unified array comparison (flattened)
                testbench_code.append(f"        if (!{var_name}_expected_json.is_array() || {var_name}_expected_json.size() < {shape[0]}) {{")
                testbench_code.append(f"            std::cerr << \"Error: expected_outputs.{var_name} is not a {array_dimension}D array (wrong number of outer elements expected: {shape[0]} has: \" << {var_name}_expected_json.size() << \").\" << std::endl;")
                testbench_code.append(f"            return 1;")
                testbench_code.append("        }")
                testbench_code.append(f"\n        // Compare array {var_name} (flattened)")
                testbench_code.append(f"        uint32_t flat_{var_name}_idx = 0;")
                testbench_code.append(f"        for (const auto& expected_val_json : {var_name}_expected_json) {{")
                testbench_code.append(f"            if (flat_{var_name}_idx >= {total_elements}) {{")
                testbench_code.append(f"                std::cerr << \"Error: expected_outputs.{var_name} JSON data has more elements than expected ({total_elements}).\" << std::endl;")
                testbench_code.append(f"                return 1;")
                testbench_code.append(f"            }}")
                testbench_code.append(f"            {cpp_base_type} expected_val = expected_val_json.get<{cpp_base_type}>();")
                testbench_code.append(f"            if (!compare_numeric({var_name}_arr[flat_{var_name}_idx], expected_val)) {{")                
                testbench_code.append(f"                if (num_mismatches < {mismatches_to_report}){{")
                testbench_code.append(f"                    std::cout << \"  FAIL: Array '{var_name}' at flattened index [\" << flat_{var_name}_idx << \"] - Expected: \" << expected_val << \", Actual: \" << {var_name}_arr[flat_{var_name}_idx] << std::endl;")
                testbench_code.append("                 }")
                testbench_code.append("                 num_mismatches++;")
                testbench_code.append("                 test_case_passed = false;")
                testbench_code.append("                 // break; // Uncomment to stop on first mismatch for this array")
                testbench_code.append("             }")
                testbench_code.append(f"            flat_{var_name}_idx++;")
                testbench_code.append("         }")
                testbench_code.append(f"        if (flat_{var_name}_idx != {total_elements}) {{")
                testbench_code.append(f"            std::cerr << \"Error: expected_outputs.{var_name} JSON data has fewer elements than expected ({total_elements}).\" << std::endl;")
                testbench_code.append(f"            return 1;")
                testbench_code.append(f"        }}")
                testbench_code.append(f"        if (num_mismatches == 0) std::cout << \"  PASS: Array '{var_name}' matches.\" << std::endl;")
    testbench_code.append("\n        if (test_case_passed) {")
    testbench_code.append("            std::cout << \"Test Case PASSED: \" << test_id << std::endl;")
    testbench_code.append("        } else {")
    testbench_code.append("            std::cout << \"Test Case FAILED: \" << test_id << std::endl;")
    testbench_code.append("            failed_tests++;")
    testbench_code.append("        }")
    # Cleanup dynamic memory for current test case
    for cleanup_line in cleanup_deallocations:
        testbench_code.append(cleanup_line)
    #Calculate the Percent Error FIXME: FInd out if we will even use this number because sometimes it is wrong and we dont even care because its just wrong.
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
        logger.info(f"Generated C++ testbench source file: {testbench_cpp_file_path}")
        return testbench_cpp_file_path, test_data_json_path, "SUCCESS", all_var_metadata
    except FileExistsError as e:
        logger.warning(f"C++ Testbench {testbench_cpp_file_path} already exists.")
        return testbench_cpp_file_path, test_data_json_path, f"SUCCESS: Testbench already exists: {e}", all_var_metadata
    except Exception as e:
        logger.error(f"Error writing C++ testbench file: {e}")
        return None, test_data_json_path, f"FAILED: Error writing C++ testbench file: {e}", all_var_metadata
    
def write_json_test_data(filename, test_cases_list):
    """
    Writes a list of test cases (inputs and expected outputs) to a JSON file.
    """
    try:
        output_dir = os.path.dirname(filename)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Created directory: {output_dir}")

        with open(filename, 'w') as f:
            json.dump(test_cases_list, f, indent=4, cls=NumpyEncoder)
        logger.info(f"Successfully wrote test data to {filename}")
    except Exception as e:
        logger.error(f"Error writing test data to {filename}: {e}")
        raise

def prepare_csim(kernel_name=None, model_dir="./results/solutions/gpt-4o/attempts_1", seed=0, hls_flow="vitis"):
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
    logger.debug(f"model_dir: {model_dir}")
    project_root = (os.path.dirname(os.path.dirname(__file__)))
    kernel_dir = os.path.join(project_root, "dataset", "kernels")
    if kernel_name is None:
        kernel_names = [d for d in os.listdir(kernel_dir)
                      if os.path.isdir(os.path.join(kernel_dir, d))]
    else:
        kernel_names = [kernel_name]
        
    all_results = {} # Changed to all_results to avoid confusion with single kernel results

    for name in kernel_names:
        logger.info(f"Processing kernel for JSON reference data: {name}")
        members = import_module_members(name, ["inputs_info","behavioral"])
        if not members or "inputs_info" not in members:
            logger.warning(f"Could not import inputs info for {name}, skipping...")
            all_results[name] = {"status": "error", "message": "Could not import inputs_info"}
            continue
        inputs_info = members["inputs_info"]
        generation_result = generate_inputs(inputs_info, np.int64(seed))
        
        if isinstance(generation_result, dict) and "inputs" in generation_result:
            inputs_list = generation_result["inputs"]
            input_types = generation_result.get("types")
        else:
            inputs_list = generation_result # Old format
            input_types = None
        logger.info(f"Generated raw inputs for {name}")
        input_arg_values_for_behavioral = []
        json_inputs_dict = {}
        json_expected_outputs_dict = {}
        output_var_names_in_order = []
        # Populate json_inputs_dict and prepare for behavioral call
        for i, type_info in enumerate(input_types):
            var_name = type_info["name"]
            current_value = inputs_list[i] # Get the actual value from the inputs_list
            if type_info["iotype"] in ["in", "inout"]:
                json_inputs_dict[var_name] = current_value
                input_arg_values_for_behavioral.append(np.copy(current_value) if isinstance(current_value, np.ndarray) else current_value)
            if type_info["iotype"] in ["out", "inout"]:
                json_inputs_dict[var_name] = current_value
                output_var_names_in_order.append(var_name)
        
        if "behavioral" in members:
            behavioral_func = members["behavioral"]
            try:
                behavioral_result_raw = behavioral_func(*input_arg_values_for_behavioral)
                logger.info(f"Behavioral model executed successfully for {name}")
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
                logger.error(f"Error running behavioral function for {name}: {e}")
                all_results[name] = {"status": "error", "message": f"Behavioral error: {e}"}
                continue
        else:
            logger.error(f"No behavioral model found for {name}, skipping JSON generation.")
            all_results[name] = {"status": "error", "message": "No behavioral model"}
            continue
        current_test_case = {
            "test_id": f"{name}_test_{seed}",
            "inputs": json_inputs_dict,
            "expected_outputs": json_expected_outputs_dict,
            "types_metadata": input_types # Include full types info for C++ to interpret easily
        }
        copy_model_dir = os.path.join(model_dir, "test_data")
        if not os.path.exists(copy_model_dir):
            os.makedirs(copy_model_dir)
            logger.debug(f"Created directory for test data: {copy_model_dir}")
        json_output_filename = os.path.abspath(os.path.join(model_dir, f"{name}_test_data.json"))
        copy_json_output_filename = os.path.abspath(os.path.join(copy_model_dir, f"{name}_test_data_copy.json"))
        try:
            write_json_test_data(json_output_filename, [current_test_case])
            write_json_test_data(copy_json_output_filename, [current_test_case]  )
            logger.info(f"Generated JSON test data for {name} at: {json_output_filename}")
        except Exception as e:
            logger.error(f"Failed to write JSON test data for {name}: {e}")
            all_results[name] = {"status": "error", "message": f"JSON write error: {e}"}
            continue
        logger.info(f"Generating C++ testbench source file for {name}")
        testbench_cpp_path, json_filename_basename, generation_status, var_metadata= create_testbench(
            kernel_name=name,
            test_data_json_path=json_output_filename, # This is the full path to the source JSON
            model_dir=model_dir,
            hls_flow=hls_flow
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

    return all_results, var_metadata

def run_csim(model_dir="./results/solutions/gpt-4o/axpy_int32/attempts_1", kernel_name=None, hls_flow="vitis"):
    logger.info("Preparing CSIM")
    result, var_metadata = prepare_csim(kernel_name, model_dir, seed=45, hls_flow=hls_flow)
    
    project_dir = model_dir
    kernel_code_path = os.path.join(project_dir, f"{kernel_name}.cpp")
    kernel_testbench_path = os.path.join(project_dir, f"{kernel_name}_tb.cpp")
    csim_elapsed_time = None
    csim_begin = dt.now()
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
        tb.cflags=-std=c++17 -fexceptions -I./include 
        csim.clean=true
        """ )
    
    sim_cmd = ["vitis-run", "--mode", "hls", "--csim", "--config", "vitis_config_csim.cfg", "--work_dir", project_dir]
    # vitis-run --mode hls --csim --config vitis_config_csim.cfg --work_dir ./results/solutions/gpt-4o/conv2d_int16/attempts_1
    # Run the vitis_hls and capture the output
    result_success = "success" in result[kernel_name]["status"]
    if result_success:
        try:
            logger.info("Running CSIM")
            sim_result = subprocess.run(
                sim_cmd,
                capture_output=True,
                check=False,
                timeout=180,  # Timeout after 3 minutes
            )
            sim_stdout = sim_result.stdout.decode('utf-8', errors='replace')  
            sim_stderr = sim_result.stderr.decode('utf-8', errors='replace') 
            csim_elapsed_time = dt.now() - csim_begin
            return sim_result, csim_elapsed_time, var_metadata
        except subprocess.TimeoutExpired as e:
            logger.warning(f"Timeout occured: {e}")
            # IMPORTANT: Access stdout and stderr from the exception object 'e'
            # Decode them just like you would from the result object
            sim_stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""
            sim_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""
            raise CsimTimeoutError(
                f"CSIM timed out after {e.timeout} seconds. Output: {sim_stdout}, Error: {sim_stderr}"
            )
        except Exception as e:
            logger.error(f"An unexpected error occurred during CSIM: {e}")
            csim_elapsed_time = dt.now() - csim_begin

            return e, csim_elapsed_time, var_metadata
    else: 
        logger.warning("No Testbench file skipping to next kernel.")
        #No Testebench was made skip this kernel. 
        csim_elapsed_time = dt.now() - csim_begin

        return "Testbench creation failed", csim_elapsed_time, var_metadata
        
def run_csynth(filepath, kernel_name, variable_metadata):
    # Add an axi or axilite bus for each of the parameters to they don't get optimized away
    axi_interfaces = ""
    var_num = 0
    for var_info in variable_metadata:
        var_name = var_info["name"]
        array_dimension = var_info["array_dimension"]
        if array_dimension == 0: # axilite for single-value variables
            next_line = f"syn.directive.interface={kernel_name} -mode s_axilite port={var_name}\n"
        else: # axi for data arrays
            next_line = f"syn.directive.interface={kernel_name} -mode m_axi -offset slave port={var_name} bundle=gmem{var_num}\n"
            var_num += 1
        axi_interfaces += next_line
    csynth_elapse_time = None
    csynth_begin = dt.now()
    with open("vitis_config.cfg", "w") as file:
        file.write(
            f"""
part=xazu1eg-sbva484-1-i

[hls]
flow_target=vivado
package.output.format=ip_catalog
package.output.syn=false
syn.top={kernel_name}
syn.file={filepath}/{kernel_name}.cpp
{axi_interfaces}
        """
        )
    try:
        result = subprocess.run(
            [
                "v++",
                "-c",
                "--mode",
                "hls",
                "--config",
                "vitis_config.cfg",
                "--work_dir",
                filepath,
            ], 
            # v++ -c --mode hls --config vitis_config.cfg --work_dir ./results/solutions/gpt-4o/...
            timeout=120,
            capture_output=True, 
            text=True
        )
        csynth_end = dt.now()
        csynth_elapse_time = csynth_end - csynth_begin
        return result, csynth_elapse_time
    except subprocess.TimeoutExpired:
        raise CsynthTimeoutError("Vitis tools stalled during HLS Synthesis.")
