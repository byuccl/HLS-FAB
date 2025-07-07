import os
import subprocess
import logging
import time
import json
from datetime import datetime
import hlseval

logger = logging.getLogger("DYNARAPID")

def run_dynarapid(kernel_name, solutions_path):
    # Extract the relevant part of the path
    # Get the path from solutions_path that matches the container's structure
    if "/results/solutions/dynarapid/" in solutions_path:
        # Extract the path starting from /results/...
        container_path = "/mount" + solutions_path[solutions_path.find("/results/solutions/dynarapid/"):]
    else:
        # Use the full path as fallback
        container_path = f"/mount{solutions_path}"
    
    logger.debug(f"Host path: {solutions_path}")
    logger.debug(f"Container path: {container_path}")
    
    # Create the TCL script content
    syn_tcl = f"""
set_project .
set_top_file {kernel_name}.cpp
synthesize -use-lsq=false -simple-buffers=true -verbose
#cdfg
write_hdl

exit
    """
    
    # Set proper umask before creating files
    # This ensures new files have appropriate permissions
    old_umask = os.umask(0)
    try:
        # Write the TCL script to a file with proper permissions from the start
        filename = f"{solutions_path}/synthesis.tcl"
        try:
            with open(filename, "w") as file:
                file.write(syn_tcl)
            logger.debug(f"Successfully wrote to {filename}")
        except Exception as e:
            logger.error(f"An error occurred while writing TCL script: {e}")
            return 1, "No feedback at this time"
        
        # Prepare the source directory with proper permissions
        try:
            os.makedirs(f"{solutions_path}/src", mode=0o777, exist_ok=True)
            os.makedirs(f"{solutions_path}/reports", mode=0o777, exist_ok=True)
            
            # Use the cp command with the --preserve=mode option to preserve mode but not ownership
            subprocess.run(f"cp {solutions_path}/{kernel_name}.cpp {solutions_path}/src/", 
                          shell=True, check=True, text=True)
            
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
            return 1, "No feedback at this time"

        # Start the container if it's not already running
        try:
            # Capture and discard the output from docker start
            result = subprocess.run("docker start dynarapid_container", 
                            shell=True, check=True, text=True, 
                            capture_output=True)  # Add capture_output=True
            logger.info("Started dynarapid_container")
            
            # Verify the container is running
            container_check = subprocess.run("docker ps | grep dynarapid_container", 
                                           shell=True, text=True, capture_output=True)
            logger.debug(f"Container status: {container_check.stdout}")
            
            if not container_check.stdout.strip():
                logger.error("Error: Container not running properly")
                return 1, "Container not running"
                
        except Exception as e:
            logger.error(f"Error with container operations: {e}")
            return 1, "Container operation failed"
        
        # Execute commands separately with timeout protection
        try:
            # Test if we can execute basic commands
            test_cmd = f"docker exec dynarapid_container bash -c 'ls -la {container_path}'"
            test_result = subprocess.run(test_cmd, shell=True, text=True, 
                                      capture_output=True, timeout=10)
            logger.debug(f"Test command output: {test_result.stdout}")
            logger.debug(f"Test command error: {test_result.stderr}")
            
            if test_result.returncode != 0:
                logger.error("Error: Cannot access the directory in the container")
                return 1, "Directory access failed"
            
            # Run the dynamatic command
            logger.info("Running dynamatic synthesis...")

            cmd = f"docker exec dynarapid_container bash -c 'cd {container_path} && dynamatic synthesis.tcl'"
            
            # Use Popen to have more control over the process
            dynamatic_begin = datetime.now()
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, text=True)


            
            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minutes timeout
                dynamatic_end = datetime.now()
                logger.debug(f"Synthesis exit code: {process.returncode}")
                logger.debug(f"Synthesis output: {stdout}")
                logger.debug(f"Synthesis error: {stderr}")
                logger.debug(f"Dynamatid elapsed time: {dynamatic_end-dynamatic_begin}")

            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("Synthesis process timed out and was killed")
                stdout, stderr = process.communicate()
                logger.debug(f"Partial output: {stdout}")
                logger.debug(f"Partial error: {stderr}")

            # Instead of chmod, use Docker to set permissions from inside the container
            # This works because Docker is likely running with sufficient privileges
            fix_perm_cmd = f"docker exec dynarapid_container bash -c 'chmod -R a+rwx {container_path}'"
            try:
                subprocess.run(fix_perm_cmd, shell=True, check=True, timeout=30)
                logger.debug("Set permissions on output files via Docker")
            except Exception as e:
                logger.error(f"Error setting permissions via Docker: {e}")
                
        except subprocess.TimeoutExpired:
            logger.warning("A command timed out")
            return 1, "Command execution timed out"
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return 1, "Unexpected error"
            
    finally:
        # Restore original umask
        os.umask(old_umask)
    
    return 0, stdout



def create_dyna_testbench(kernel_name, test_data_json_path, model_dir):
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
    logger.info("Generating H File for CSIM")
    hlseval.generate_h_file(kernel_code, model_dir, kernel_name)
    
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
    testbench_code.append("#include \"nlohmann/apint_serializer.hpp\"")    # ap_int serializer

    # Include the actual kernel file 
    testbench_code.append(f"#include \"{kernel_name}.h\"\n\n")


    testbench_code.append("using json = nlohmann::json;")
    testbench_code.append("using namespace std;\n\n")

    testbench_code.append("int main() {")

    # Open and parse JSON test data
    # NOTE: The JSON file path needs to be accessible from where the C++ executable runs.
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
        in_out_status = var_info["in_out"]
        dtype_str = var_info["dtype"] # e.g., "int32", "float64", "int", "float"
        cpp_base_type = hlseval.get_norm_cpp_type_str(dtype_str)
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
        elif array_dimension >= 1:
            # Declare and allocate flattened array
            testbench_code.append(f"        {cpp_base_type}* {var_name}_arr = new {cpp_base_type}[{total_elements}];")
            cleanup_deallocations.append(f"        delete[] {var_name}_arr;")

            # Initialize from JSON for inputs
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

                testbench_code.append(f"        // In testbench, after loading {var_name}_arr")
                testbench_code.append(f"        std::cout << \"{var_name} (first 10 elements): \";")
                testbench_code.append(f"        for (int i = 0; i < 10; ++i) {{")
                testbench_code.append(f"            std::cout << {var_name}_arr[i] << \" \";")
                testbench_code.append(f"        }}")
                testbench_code.append(f"        std::cout << std::endl;")      
                
            elif in_out_status == "out":
                # Initialize output buffers to 0 for clarity
                testbench_code.append(f"\n        // Initialize output array {var_name}_arr to zeros")
                testbench_code.append(f"        for (int i = 0; i < {total_elements}; ++i) {{")
                testbench_code.append(f"            {var_name}_arr[i] = 0; // Or appropriate default value")
                testbench_code.append("        }")

            # Add to kernel arguments (passing pointer to flattened array)
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
        cpp_base_type = hlseval.get_norm_cpp_type_str(dtype_str)
        shape = var_info.get('shape', []) # Shape from metadata (e.g., [10, 20])
        total_elements = var_info.get('total_elements', 1) # Total elements for flattening
        
        if in_out_status in ["out", "inout"]:
            # Get expected values from JSON
            testbench_code.append(f"\n        num_mismatches = 0;")
            
            testbench_code.append(f"\n        // Get expected values for {var_name}")
            testbench_code.append(f"        auto {var_name}_expected_json = test_case.at(\"expected_outputs\").at(\"{var_name}\");")
            
            # Check for the outer-most dimension
            if array_dimension > 0: # Only check for arrays
                testbench_code.append(f"        if (!{var_name}_expected_json.is_array() || {var_name}_expected_json.size() < {shape[0]}) {{")
                testbench_code.append(f"            std::cerr << \"Error: expected_outputs.{var_name} is not a {array_dimension}D array (wrong number of outer elements expected: {shape[0]} has: \" << {var_name}_expected_json.size() << \").\" << std::endl;")
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


            elif array_dimension >= 1: # Unified array comparison (flattened)
                testbench_code.append(f"\n        // Compare array {var_name} (flattened)")
                testbench_code.append(f"        uint32_t flat_compare_idx = 0;")
                testbench_code.append(f"        for (const auto& expected_val_json : {var_name}_expected_json) {{")
                testbench_code.append(f"            if (flat_compare_idx >= {total_elements}) {{")
                testbench_code.append(f"                std::cerr << \"Error: expected_outputs.{var_name} JSON data has more elements than expected ({total_elements}).\" << std::endl;")
                testbench_code.append(f"                return 1;")
                testbench_code.append(f"            }}")
                testbench_code.append(f"            {cpp_base_type} expected_val = expected_val_json.get<{cpp_base_type}>();")
                testbench_code.append(f"            if (!compare_numeric({var_name}_arr[flat_compare_idx], expected_val)) {{")                
                testbench_code.append(f"                if (num_mismatches < {mismatches_to_report}){{")
                testbench_code.append(f"                    std::cout << \"  FAIL: Array '{var_name}' at flattened index [\" << flat_compare_idx << \"] - Expected: \" << expected_val << \", Actual: \" << {var_name}_arr[flat_compare_idx] << std::endl;")
                testbench_code.append("                 }")
                testbench_code.append("                 num_mismatches++;")
                testbench_code.append("                 test_case_passed = false;")
                testbench_code.append("                 // break; // Uncomment to stop on first mismatch for this array")
                testbench_code.append("             }")
                testbench_code.append(f"            flat_compare_idx++;")
                testbench_code.append("         }")
                testbench_code.append(f"        if (flat_compare_idx != {total_elements}) {{")
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
        logger.info(f"Generated C++ testbench source file: {testbench_cpp_file_path}")
        return testbench_cpp_file_path, test_data_json_path, "SUCCESS"
    except FileExistsError as e:
        logger.warning(f"C++ Testbench {testbench_cpp_file_path} already exists.")
        return testbench_cpp_file_path, test_data_json_path, f"SUCCESS: Testbench already exists: {e}"
    except Exception as e:
        logger.error(f"Error writing C++ testbench file: {e}")
        return None, test_data_json_path, f"FAILED: Error writing C++ testbench file: {e}"

