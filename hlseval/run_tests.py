import os
import json
import numpy as np
import argparse  
from hlseval import run_cpp_with_inputs, import_module_functions



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



def test_kernel(kernel_name=None, model_dir="gpt-4o/attempts_1", seed=0):    
    """    
    Test a kernel by generating inputs and running the C++ implementation.    
    """    
    # Get the list of kernel directories    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))    
    kernel_dir = os.path.join(project_root, "dataset", "kernels")    
        
    # If kernel_name is not provided, get all directories in the kernels folder    
    if kernel_name is None:    
        kernel_names = [d for d in os.listdir(kernel_dir)     
                      if os.path.isdir(os.path.join(kernel_dir, d))]    
    else:    
        kernel_names = [kernel_name]    
        
    results = {}    
        
    for name in kernel_names:    
        # Initialize result validation variables    
        match_status = "unknown"    
        match_details = None    
        testbench_info = ""  

        print(f"\n--- Testing kernel: {name} ---")    
            
        # Import the functions    
        functions = import_module_functions(name, ["generate_inputs", "behavioral"])    

        if not functions or "generate_inputs" not in functions:    
            print(f"Could not import generate_inputs for {name}, skipping...")    
            results[name] = {"status": "error", "message": "Could not import generate_inputs"}    
            continue    
            
        # Generate inputs    
        generate_inputs_func = functions["generate_inputs"]    
        generation_result = generate_inputs_func(np.int64(seed))  
          
        # Handle both old and new formats  
        if isinstance(generation_result, dict) and "inputs" in generation_result:  
            inputs = generation_result["inputs"]  
            input_types = generation_result.get("types")  
        else:  
            # For backward compatibility with the old format  
            inputs = generation_result  
            input_types = None  
              
        print(f"Generated inputs for {name}")    
            
        # Create a copy of inputs for the behavioral model    
        behavioral_inputs = [np.copy(inp) if isinstance(inp, np.ndarray) else inp for inp in inputs]    
        rtps = [inp for inp in inputs if not isinstance(inp, np.ndarray) ]
        just_inputs = [
            arr for arr, type_info in zip(behavioral_inputs, input_types)
            if type_info.get("in_out") == "in"
        ]
        # Determine output matrix index from input types if available  
        output_matrix_indices = []  
        if input_types:  
            for i, type_info in enumerate(input_types):  
                if type_info.get('in_out') == 'out':  
                    output_matrix_indices.append(i) 
                    print(f"Found output at index {i} from type information")  
                    # break  
          
        # If available, run the behavioral model for reference    
        expected_result = None    
        
        if "behavioral" in functions:    
            behavioral_func = functions["behavioral"]    
            try:    
                # Run behavioral function and get expected output    
                behavioral_result = behavioral_func(*just_inputs)   
                print(f"Behavioral model executed successfully")    
                    
                # Determine if behavioral function returns a result or modifies an input    
                if behavioral_result is not None:    
                    # Direct return value case    
                    expected_result = behavioral_result   
                    if isinstance(expected_result, tuple):
                        print("Behavioral function returned a tuple") 
                    else:
                        print(f"Behavioral function returned a result of shape {expected_result.shape}")    
                else:    
                    # Input modification case - find which input was modified    
                    for i, (orig_inp, modified_inp) in enumerate(zip(inputs, behavioral_inputs)):    
                        if isinstance(orig_inp, np.ndarray) and isinstance(modified_inp, np.ndarray):    
                            if not np.array_equal(orig_inp, modified_inp):    
                                expected_result = modified_inp    
                                if output_matrix_indices is None:  
                                    output_matrix_indices.append(i)    
                                print(f"Behavioral function modified input {i} to produce the result")    
                                break    
                    
                if expected_result is None:    
                    print("Warning: Could not determine expected output from behavioral function")    
            except Exception as e:    
                print(f"Error running behavioral function: {e}")    
            
        # If we didn't determine output matrix index yet, try to infer it    
        if output_matrix_indices is None:    
            # Look for zero-initialized arrays (common for output buffers)    
            for i, inp in enumerate(inputs):    
                if isinstance(inp, np.ndarray) and np.all(inp == 0):    
                    output_matrix_indices.append(i)  
                    print(f"Inferred output matrix to be input {i} (initialized to zeros)")    
                    break    
                
            # If still not found, assume last array is output    
            if output_matrix_indices is None:    
                for i in range(len(inputs) - 1, -1, -1):    
                    if isinstance(inputs[i], np.ndarray):    
                        output_matrix_indices.append(i)  
                        print(f"Assuming output matrix is input {i} (last array)")    
                        break    
            
        # Run the C++ implementation, passing the identified output_matrix_index    
        return_code, stdout, stderr, cpp_result_matrix = run_cpp_with_inputs(    
            name, inputs, input_types, model_dir, output_matrix_indices 
        )    
            


            
        if return_code == 0:    
            print(f"C++ implementation ran successfully")    
                
            # Validate the result if we have both expected and actual results      
            if expected_result is not None and cpp_result_matrix is not None:  #FIXME: We want to check all outputs expected_results could be a tuple
                    # Support tuple of outputs or single output
                expected_list = expected_result if isinstance(expected_result, tuple) else (expected_result,)
                actual_list = cpp_result_matrix if isinstance(cpp_result_matrix, tuple) else (cpp_result_matrix,)
                
                if len(expected_list) != len(actual_list):
                    print(f"Mismatch in number of outputs: expected {len(expected_list)}, got {len(actual_list)}")
                else:
                    for idx, (expected_output, actual_output) in enumerate(zip(expected_list, actual_list)):
                        print(f"\nValidating output #{idx + 1}:")
                
                        # Determine if we're working with floating point values  
                        is_float_type = (hasattr(expected_output, 'dtype') and np.issubdtype(expected_output.dtype, np.floating)) or (hasattr(actual_output, 'dtype') and np.issubdtype(actual_output.dtype, np.floating)) or isinstance(expected_output, float) or isinstance(actual_output, float)  
                        
                        # Set appropriate tolerance values based on data type  
                        rtol = 1e-5  # relative tolerance  
                        atol = 1e-8  # absolute tolerance  
                        if is_float_type:  
                            # Use looser tolerances for float32 (vs float64)  
                            if hasattr(expected_output, 'dtype') and expected_output.dtype == np.float32:  
                                rtol = 1e-4  
                                atol = 1e-6  
                        
                        # Handle scalar comparison differently  
                        if (np.isscalar(expected_output) or (isinstance(expected_output, np.ndarray) and expected_output.size == 1)) and (np.isscalar(actual_output) or (isinstance(actual_output, np.ndarray) and actual_output.size == 1)):  
                            # Extract scalar values  
                            expected_scalar = expected_output.item() if hasattr(expected_output, 'item') else expected_output  
                            actual_scalar = actual_output.item() if hasattr(actual_output, 'item') else actual_output  
                            
                            # Compare with tolerance for floating point values  
                            if is_float_type:  
                                match = np.isclose(expected_scalar, actual_scalar, rtol=rtol, atol=atol)  
                            else:  
                                match = expected_scalar == actual_scalar  
                            
                            if match:  
                                match_status = "match"  
                                print("C++ implementation result matches expected output")  
                            else:  
                                match_status = "mismatch"  
                                # Calculate differences  
                                abs_diff = abs(expected_scalar - actual_scalar)  
                                rel_diff = abs_diff / (abs(expected_scalar) + 1e-10)  
                                
                                match_details = {  
                                    "mismatch_count": 1,  
                                    "total_elements": 1,  
                                    "mismatch_percentage": 100.0,  
                                    "max_absolute_diff": float(abs_diff),  
                                    "mean_absolute_diff": float(abs_diff),  
                                    "max_relative_diff": float(rel_diff),  
                                    "mean_relative_diff": float(rel_diff)  
                                }  
                                testbench_info = f"""
Results mismatch: Scalar values differ
\tExpected: {expected_scalar}, Actual: {actual_scalar}
\tAbsolute difference: {abs_diff}
\tRelative difference: {rel_diff}
\tTolerances used: rtol={rtol}, atol={atol}
                                """
                                
                        else:  
                            # Array comparison with tolerance for floating point  
                            if is_float_type:  
                                # Use np.isclose for element-wise comparison with tolerance  
                                matches = np.isclose(expected_output, actual_output, rtol=rtol, atol=atol)  
                                match_percentage = np.mean(matches) * 100  
                                if match_percentage == 100:  
                                    match_status = "match"  
                                    print("C++ implementation result matches expected output (within floating-point tolerance)")  
                                else:  
                                    match_status = "mismatch"  
                                    
                                    # Calculate how many elements don't match  
                                    mismatch_count = np.sum(~matches)  
                                    total_elements = expected_output.size  
                                    mismatch_percentage = (mismatch_count / total_elements) * 100  
                                    
                                    # Calculate detailed statistics on differences  
                                    abs_diff = np.abs(expected_output - actual_output)  
                                    max_abs_diff = np.max(abs_diff)  
                                    mean_abs_diff = np.mean(abs_diff)  
                                    
                                    # Handle division by zero in relative difference calculation  
                                    with np.errstate(divide='ignore', invalid='ignore'):  
                                        rel_diff = abs_diff / (np.abs(expected_output) + 1e-10)  
                                        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)  
                                    max_rel_diff = np.max(rel_diff)  
                                    mean_rel_diff = np.mean(rel_diff)  
                                    
                                    match_details = {  
                                        "mismatch_count": int(mismatch_count),  
                                        "total_elements": int(total_elements),  
                                        "mismatch_percentage": float(mismatch_percentage),  
                                        "max_absolute_diff": float(max_abs_diff),  
                                        "mean_absolute_diff": float(mean_abs_diff),  
                                        "max_relative_diff": float(max_rel_diff),  
                                        "mean_relative_diff": float(mean_rel_diff),  
                                        "tolerances": {"rtol": rtol, "atol": atol}  
                                    }  
                                    
                                    print(f"Results exceed floating-point tolerance: {mismatch_count}/{total_elements} elements ({mismatch_percentage:.2f}%)")  
                                    print(f"   Max absolute difference: {max_abs_diff}")  
                                    print(f"   Mean absolute difference: {mean_abs_diff}")  
                                    print(f"   Tolerances used: rtol={rtol}, atol={atol}")  
                                    
                                    # Output sample mismatches  
                                    if mismatch_count > 0:  
                                        testbench_info += "\nSample mismatches (expected vs. actual):\n"  
                                        mismatch_indices = np.where(~matches)  
                                        for i in range(min(5, len(mismatch_indices[0]))):  
                                            idx = tuple(coord[i] for coord in mismatch_indices)  
                                            testbench_info += f"  At position {idx}: {expected_output[idx]} vs {actual_output[idx]}\n"  
                            else:  
                                # For non-floating point types, use exact equality      
                                if np.array_equal(expected_output, actual_output):      
                                    match_status = "match"      
                                    print("C++ implementation result matches expected output")      
                                else:    
                                    match_status = "mismatch"      
            
                                    # Calculate how many elements don't match      
                                    mismatch_mask = expected_output != actual_output  
                                    mismatch_count = np.sum(mismatch_mask)      
                                    total_elements = expected_output.size      
                                    mismatch_percentage = (mismatch_count / total_elements) * 100      
                                    
                                    # Calculate detailed statistics on differences for integers  
                                    abs_diff = np.abs(expected_output.astype(np.int64) - actual_output.astype(np.int64))  
                                    max_abs_diff = np.max(abs_diff)  
                                    mean_abs_diff = np.mean(abs_diff)  
                                    
                                    # Handle division by zero in relative difference calculation  
                                    with np.errstate(divide='ignore', invalid='ignore'):  
                                        rel_diff = abs_diff / (np.abs(expected_output.astype(np.float64)) + 1e-10)  
                                        rel_diff = np.where(np.isfinite(rel_diff), rel_diff, 0)  
                                    max_rel_diff = np.max(rel_diff)  
                                    mean_rel_diff = np.mean(rel_diff)  
                                    
                                    match_details = {  
                                        "mismatch_count": int(mismatch_count),  
                                        "total_elements": int(total_elements),  
                                        "mismatch_percentage": float(mismatch_percentage),  
                                        "max_absolute_diff": int(max_abs_diff),  
                                        "mean_absolute_diff": float(mean_abs_diff),  
                                        "max_relative_diff": float(max_rel_diff),  
                                        "mean_relative_diff": float(mean_rel_diff)  
                                    }  
                                    
                                    print(f"Results mismatch: {mismatch_count}/{total_elements} elements ({mismatch_percentage:.2f}%)")  
                                    print(f"   Max absolute difference: {max_abs_diff}")  
                                    print(f"   Mean absolute difference: {mean_abs_diff}")  
                                    
                                    # Output sample mismatches  
                                    if mismatch_count > 0:  
                                        testbench_info = "\nSample mismatches (expected vs. actual):\n"  
                                        mismatch_indices = np.where(mismatch_mask)  
                                        for i in range(min(5, len(mismatch_indices[0]))):  
                                            idx = tuple(coord[i] for coord in mismatch_indices)  
                                            testbench_info += f"  At position {idx}: {expected_output[idx]} vs {actual_output[idx]}\n"  
                                        else:    
                                            if expected_result is None:    
                                                print("No expected result available for comparison")    
                                            if cpp_result_matrix is None:    
                                                print("No C++ result matrix could be extracted from output")    
                                        
            # Store results    
            results[name] = {    
                "status": "success",    
                "match_status": match_status,    
                "match_details": match_details, 
                "testbench_info": testbench_info,
                "output_matrix_index": output_matrix_indices,    
                "stdout": stdout    
            }    
                
            # Include the matrices in results (convert to lists for JSON serialization)    
            if cpp_result_matrix is not None:
                results[name]["cpp_result"] = (cpp_result_matrix.tolist() if isinstance(cpp_result_matrix, np.ndarray) else list(cpp_result_matrix))
            if expected_result is not None: 
                if isinstance(expected_result, tuple):
                    results[name]["expected_result"] = list(expected_result)
                elif isinstance(expected_result, np.ndarray):
                    results[name]["expected_result"] = expected_result.tolist()
                else:
                    results[name]["expected_result"] = [expected_result]
                    
        else:    
            print(f"C++ implementation failed with code {return_code}")    
            if stderr:    
                print(f"Error: {stderr}")    
            results[name] = {    
                "status": "failure",    
                "match_status": "C++ implementation failed to compile. Unable to compare results to expected ouptut.",    
                "testbench_info": stderr if stderr else "",    
                "return_code": return_code,    
                "stderr": stderr if stderr else "",    
                "stdout": stdout if stdout else ""    
            }    
        
    return results  