import os
import numpy as np
import re  
import tempfile  
import subprocess  
import traceback  
from hlseval import get_cpp_type



# Helper function to extract matrices from output  
def extract_matrix_from_output(output, matrix_name, dimensions, shape, dtype):  
    """Extract a matrix of arbitrary dimension from the output text"""  
  
    if not output:  
        print(f"No output provided to extract matrix {matrix_name}")  
        return None  
  
    # Handle scalar (0-dimensional) case specially  
    if dimensions == 0:  
        scalar_pattern = f"{matrix_name} scalar value: (.*?)[\r\n]"  
        match = re.search(scalar_pattern, output)  
  
        if match:  
            try:  
                value_str = match.group(1).strip()  
                if dtype.kind in 'iu':  # Integer types  
                    return np.array(int(value_str), dtype=dtype)  
                else:  # Float types  
                    return np.array(float(value_str), dtype=dtype)  
            except (ValueError, IndexError) as e:  
                print(f"Error extracting scalar value: {e}")  
  
        pattern = f"{matrix_name} values after kernel execution:"  
        parts = output.split(pattern)  
        if len(parts) >= 2:  
            try:  
                text = parts[1].strip().split('\n')[0]  
                for word in text.split():  
                    try:  
                        if dtype.kind in 'iu':  # Integer types  
                            return np.array(int(word), dtype=dtype)  
                        else:  # Float types  
                            return np.array(float(word), dtype=dtype)  
                    except ValueError:  
                        continue  
            except Exception as e:  
                print(f"Error extracting scalar from text: {e}")  
  
        print("Could not extract scalar value using standard methods, attempting backup methods...")  
  
        possible_patterns = [  
            f"{matrix_name}\\s*=\\s*([+-]?\\d+(?:\\.\\d+)?)",  
            f"{matrix_name}\\s*:\\s*([+-]?\\d+(?:\\.\\d+)?)",  
            f"{matrix_name}.*?([+-]?\\d+(?:\\.\\d+)?)"  
        ]  
  
        for pattern in possible_patterns:  
            matches = re.findall(pattern, output)  
            if matches:  
                try:  
                    value_str = matches[0].strip()  
                    if dtype.kind in 'iu':  # Integer types  
                        return np.array(int(value_str), dtype=dtype)  
                    else:  # Float types  
                        return np.array(float(value_str), dtype=dtype)  
                except (ValueError, IndexError):  
                    continue  
  
        print(f"Failed to extract scalar value for {matrix_name}")  
        return None  
  
    # Initialize the result array and ensure it's overwritten correctly  
    result = np.zeros(shape, dtype=dtype)  
  
    print(f"Initializing result array for {matrix_name} with shape {shape} and dtype {dtype}")  
  
    start_marker = f"START_DATA_{matrix_name}"  
    end_marker = f"END_DATA_{matrix_name}"  
  
    start_pos = output.find(start_marker)  
    if start_pos == -1:  
        print(f"Could not find start marker '{start_marker}' in output")  
  
        pattern = f"{matrix_name} values after kernel execution:"  
        parts = output.split(pattern)  
        if len(parts) < 2:  
            print(f"Could not find matrix {matrix_name} in output using fallback method")  
            return None  
  
        values_text = parts[1].split("\n")  
        values = []  
  
        for line in values_text[1:]:  
            line = line.strip()  
            if not line or line.startswith("Kernel") or line.startswith("input_"):  
                break  
  
            for val_str in line.split():  
                try:  
                    values.append(float(val_str))  
                except ValueError:  
                    continue  
  
        for i in range(min(len(values), shape[0])):  
            result[i] = values[i]  
  
        return result.astype(dtype)  
  
    end_pos = output.find(end_marker, start_pos)  
    if end_pos == -1:  
        print(f"Could not find end marker '{end_marker}' in output")  
        return None  
  
    data_section = output[start_pos + len(start_marker):end_pos].strip()  
  
    try:  
        skip_tokens = {"Hyperslice", "Slice", ":"}  
  
        if dimensions == 1:  
            values = []  
  
            for val_str in data_section.split():  
                try:  
                    num_val = int(val_str) if dtype.kind in 'iu' else float(val_str)  
                    values.append(num_val)  
                except ValueError:  
                    print(f"Warning: Could not convert '{val_str}' to number, skipping")  
  
            for i in range(min(len(values), shape[0])):  
                result[i] = values[i]  
  
        elif dimensions == 2:  
            lines = data_section.strip().split('\n')  
            row = 0  
  
            for line in lines:  
                if not line:  
                    continue  
  
                cols = []  
                for val_str in line.split():  
                    try:  
                        num_val = int(val_str) if dtype.kind in 'iu' else float(val_str)  
                        cols.append(num_val)  
                    except ValueError:  
                        print(f"Warning: Could not convert '{val_str}' to number, skipping")  
  
                for col, val in enumerate(cols):  
                    if col < shape[1]:  
                        result[row, col] = val  
  
                row += 1  
                if row >= shape[0]:  
                    break  
  
        elif dimensions == 3:  
            result = extract_3d_array_improved(output, matrix_name, shape, dtype)  

  
        elif dimensions == 4:  
            result = extract_4d_array_improved(output, matrix_name, shape, dtype)
  
        print(f"Successfully extracted matrix {matrix_name} with final shape {result.shape}")  
        return result.astype(dtype)  
  
    except Exception as e:  
        print(f"Error extracting matrix from output: {e}")  
        traceback.print_exc()  
        return None  
    
def extract_3d_array_improved(output, matrix_name, shape, dtype):  
    """Extract a 3D matrix with better handling of structured format"""  
      
    # Initialize the result array  
    result = np.zeros(shape, dtype=dtype)  
      
    # Find the structured format markers  
    start_marker = f"START_DATA_{matrix_name}"  
    end_marker = f"END_DATA_{matrix_name}"  
      
    start_pos = output.find(start_marker)  
    if start_pos == -1:  
        print(f"Could not find start marker '{start_marker}' in output")  
        return None  
          
    end_pos = output.find(end_marker, start_pos)  
    if end_pos == -1:  
        print(f"Could not find end marker '{end_marker}' in output")  
        return None  
      
    # Extract data section  
    data_section = output[start_pos + len(start_marker):end_pos].strip()  
      
    # Split into blocks by slice markers or double newlines  
    blocks = []  
    current_block = []  
      
    for line in data_section.split('\n'):  
        line = line.strip()  
          
        if line.startswith("Slice"):  
            if current_block:  
                blocks.append(current_block)  
                current_block = []  
            continue  
          
        if line:  # Skip empty lines  
            current_block.append(line)  
      
    # Add the last block if it exists  
    if current_block:  
        blocks.append(current_block)  
      
    # Process each block  
    for dim1_idx, block in enumerate(blocks):  
        if dim1_idx >= shape[0]:  
            break  
              
        for dim2_idx, line in enumerate(block):  
            if dim2_idx >= shape[1]:  
                break  
                  
            # Parse values from this line  
            values = []  
            for val_str in line.split():  
                try:  
                    # Parse as int or float based on dtype  
                    num_val = int(val_str) if dtype.kind in 'iu' else float(val_str)  
                    values.append(num_val)  
                except ValueError:  
                    continue  
                      
            # Store values in result array  
            for dim3_idx, val in enumerate(values):  
                if dim3_idx < shape[2]:  
                    result[dim1_idx, dim2_idx, dim3_idx] = val  
      
    return result  

def extract_4d_array_improved(output, matrix_name, shape, dtype):  
    """Extract a 4D array using a robust algorithm that handles structured format"""  
      
    # Initialize the result array  
    result = np.zeros(shape, dtype=dtype)  
      
    # Find the structured format markers  
    start_marker = f"START_DATA_{matrix_name}"  
    end_marker = f"END_DATA_{matrix_name}"  
      
    start_pos = output.find(start_marker)  
    if start_pos == -1:  
        print(f"Could not find start marker '{start_marker}' in output")  
        return None  
          
    end_pos = output.find(end_marker, start_pos)  
    if end_pos == -1:  
        print(f"Could not find end marker '{end_marker}' in output")  
        return None  
      
    # Extract data section  
    data_section = output[start_pos + len(start_marker):end_pos].strip()  
      
    # Extract all numeric values  
    numeric_values = []  
      
    # Split by lines  
    for line in data_section.split('\n'):  
        line = line.strip()  
          
        # Skip non-data lines  
        if not line or not any(c.isdigit() or c == '-' for c in line):  
            continue  
              
        # Process each value in the line  
        for val_str in line.split():  
            try:  
                # Skip if it's not a number  
                if not any(c.isdigit() or c == '-' for c in val_str):  
                    continue  
                      
                # Convert to the appropriate type  
                if dtype.kind in 'iu':  # Integer type  
                    numeric_values.append(int(val_str))  
                else:  # Float type  
                    numeric_values.append(float(val_str))  
            except ValueError:  
                continue  
      
    # Limit to the maximum number of elements in the array  
    max_elements = shape[0] * shape[1] * shape[2] * shape[3]  
    numeric_values = numeric_values[:max_elements]  
      
    # Fill the array in the correct shape  
    idx = 0  
    for n in range(shape[0]):  
        for c in range(shape[1]):  
            for h in range(shape[2]):  
                for w in range(shape[3]):  
                    if idx < len(numeric_values):  
                        result[n, c, h, w] = numeric_values[idx]  
                        idx += 1  
                    else:  
                        # We've run out of values  
                        return result  
      
    return result  

def run_cpp_with_inputs(kernel_name, inputs, input_types=None, model_dir="gpt-4o/", output_matrix_indices=None):      
    """      
    Compile and run a C++ function with the given inputs,      
    with support for arrays up to 4D.      
          
    Args:      
        kernel_name (str): Name of the kernel      
        inputs (list/tuple): Inputs to pass to the C++ function      
        input_types (list, optional): List of dictionaries with type information for each input      
        model_dir (str): Directory name within results/solutions/ containing the kernel files      
        output_matrix_index (int, optional): Index of the output(s) matrix in inputs      
              
    Returns:      
        tuple: (returncode, stdout, stderr, result_matrix)      
    """      

      
    
    # Get project root and construct paths      
    project_root = os.path.dirname(os.path.dirname(__file__))      
    cpp_file = os.path.join(project_root, "results", "solutions", model_dir, f"{kernel_name}.cpp")      
          
    if not os.path.exists(cpp_file):      
        print(f"C++ file not found: {cpp_file}")      
        return None, None, None, None      
          
    # Track array information  
    array_info = {}  
      
    # Determine output matrix index if not provided  
    if input_types and output_matrix_indices is None:    
        for i, type_info in enumerate(input_types):    
            if type_info.get('in_out') == 'out':    
                output_matrix_indices.append(i)    
                print(f"Using output matrix at index {i} based on type information")    
                break    
        
    # If still no output_matrix_index, try to infer from inputs  
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
      
    # Create a temporary directory for compilation and execution  
    with tempfile.TemporaryDirectory() as temp_dir:  
        # Read the kernel code  
        try:  
            with open(cpp_file, 'r', encoding='utf-8') as f:  
                kernel_code = f.read()  
        except UnicodeDecodeError:  
            # Try with a different encoding if UTF-8 fails  
            with open(cpp_file, 'r', encoding='latin-1') as f:  
                kernel_code = f.read()  
              
        # Prepare code sections  
        var_declarations = []  
        array_allocations = []  
        array_initializations = []  
        kernel_args = []  
        cleanup_code = []  
          
        # Process each input    
        for i, input_val in enumerate(inputs):    
            var_name = f"input_{i}"    
            type_info = input_types[i] if input_types and i < len(input_types) else None    
            dimension = type_info.get('dimension', None) if type_info else None    
            cpp_type = get_cpp_type(input_val, type_info)    
                
            # Store array information for later extraction    
            if isinstance(input_val, np.ndarray):    
                array_info[var_name] = {    
                    'shape': input_val.shape,    
                    'dtype': input_val.dtype,    
                    'dimension': dimension if dimension is not None else len(input_val.shape)    
                }    
                
            # Handle scalar (dimension 0)    
            if dimension == 0 or (not isinstance(input_val, np.ndarray) and not dimension):  
                scalar_value = input_val.item() if isinstance(input_val, np.ndarray) else input_val  
                base_type = cpp_type  
                
                # For scalar outputs, add them to array_info for extraction  
                if type_info and type_info.get('in_out') == 'out':  
                    array_info[var_name] = {  
                        'shape': (),  # Empty tuple for scalar  
                        'dtype': np.dtype(type(scalar_value)),  
                        'dimension': 0  
                    }
                    print(f"Scalar output detected for {var_name}, type: {np.dtype(type(scalar_value))}")
                    var_declarations.append(f"    {base_type}* {var_name} = new {base_type};")    #Treat scalar as pointer if output
                    kernel_args.append(var_name)
                    cleanup_code.append(f"  delete[]{var_name};")
                    continue
                # This is the fix - declare scalar variables with their actual type:  
                var_declarations.append(f"    {base_type} {var_name} = {scalar_value};")  
                kernel_args.append(var_name)  
                continue    
              
            # For arrays, determine base type without the pointers  
            base_type = cpp_type  
            while base_type.endswith('*'):  
                base_type = base_type[:-1]  
              
            # Handle arrays based on their dimension  
            if isinstance(input_val, np.ndarray):  
                # Get shape information  
                shape = input_val.shape  
                actual_dim = len(shape)  
                  
                # Use specified dimension if provided, otherwise use actual dimension  
                dim_to_use = dimension if dimension is not None else actual_dim  
                  
                # Handle each dimensionality  
                if dim_to_use == 1:  # 1D array  
                    size = shape[0] if actual_dim > 0 else 0  
                    size_var = f"{var_name}_size"  
                      
                    var_declarations.append(f"    const int {size_var} = {size};")  
                    array_allocations.append(f"    {base_type}* {var_name} = new {base_type}[{size}];")     

                    # Initialize values  
                    init_values = []  
                    if actual_dim == 1:  
                        for j in range(size):  
                            value = input_val[j]  
                            if input_val.dtype.kind in 'iu':  # Integer  
                                init_values.append(f"    {var_name}[{j}] = {int(value)};")  
                            else:  # Float  
                                init_values.append(f"    {var_name}[{j}] = {float(value)};")  
                    elif actual_dim > 1:  
                        # Flatten higher dimensional array  
                        flat_array = input_val.flatten()  
                        for j in range(min(size, len(flat_array))):  
                            value = flat_array[j]  
                            if input_val.dtype.kind in 'iu':  
                                init_values.append(f"    {var_name}[{j}] = {int(value)};")  
                            else:  
                                init_values.append(f"    {var_name}[{j}] = {float(value)};")  
                      
                    array_initializations.append("\n".join(init_values))  
                    #Clean Up the allocated arrays
                    cleanup_code.append(f"    delete[] {var_name};")  
                  
                elif dim_to_use == 2:  # 2D array  
                    rows = shape[0] if actual_dim > 0 else 0  
                    cols = shape[1] if actual_dim > 1 else 0  
                    rows_var = f"{var_name}_rows"  
                    cols_var = f"{var_name}_cols"  
                      
                    var_declarations.append(f"    const int {rows_var} = {rows};")  
                    var_declarations.append(f"    const int {cols_var} = {cols};")  
                      
                    # Allocate row pointers  
                    array_allocations.append(f"    {base_type} {var_name}[{rows_var}][{cols_var}];")  
                    
                    # Initialize values  
                    init_values = []  
                    if actual_dim == 2:  
                        for r in range(rows):  
                            for c in range(cols):  
                                value = input_val[r, c]  
                                if input_val.dtype.kind in 'iu':  
                                    init_values.append(f"    {var_name}[{r}][{c}] = {int(value)};")  
                                else:  
                                    init_values.append(f"    {var_name}[{r}][{c}] = {float(value)};")  
                    elif actual_dim == 1:  
                        # Stretch 1D array into 2D  
                        for r in range(rows):  
                            for c in range(cols):  
                                idx = r * cols + c  
                                if idx < len(input_val):  
                                    value = input_val[idx]  
                                    if input_val.dtype.kind in 'iu':  
                                        init_values.append(f"    {var_name}[{r}][{c}] = {int(value)};")  
                                    else:  
                                        init_values.append(f"    {var_name}[{r}][{c}] = {float(value)};")  
                                else:  
                                    init_values.append(f"    {var_name}[{r}][{c}] = 0;")  
                    elif actual_dim > 2:  
                        # Flatten first two dimensions  
                        for r in range(rows):  
                            for c in range(cols):  
                                if r < input_val.shape[0] and c < input_val.shape[1]:  
                                    if actual_dim == 3:  
                                        # For 3D, use first element of 3rd dimension  
                                        value = input_val[r, c, 0]  
                                    else:  
                                        # For 4D+, use first element of higher dimensions  
                                        value = input_val[(r, c) + (0,) * (actual_dim - 2)]  
                                      
                                    if input_val.dtype.kind in 'iu':  
                                        init_values.append(f"    {var_name}[{r}][{c}] = {int(value)};")  
                                    else:  
                                        init_values.append(f"    {var_name}[{r}][{c}] = {float(value)};")  
                                else:  
                                    init_values.append(f"    {var_name}[{r}][{c}] = 0;")  
                      
                    array_initializations.append("\n".join(init_values))  
                     
                  
                elif dim_to_use == 3:  # 3D array  
                    dim1 = shape[0] if actual_dim > 0 else 0  
                    dim2 = shape[1] if actual_dim > 1 else 0  
                    dim3 = shape[2] if actual_dim > 2 else 0  
                    dim1_var = f"{var_name}_dim1"  
                    dim2_var = f"{var_name}_dim2"  
                    dim3_var = f"{var_name}_dim3"  
                      
                    var_declarations.append(f"    const int {dim1_var} = {dim1};")  
                    var_declarations.append(f"    const int {dim2_var} = {dim2};")  
                    var_declarations.append(f"    const int {dim3_var} = {dim3};")  
                      
                    # Allocate 3D array  
                    array_allocations.append(f"    {base_type} {var_name}[{dim1_var}][{dim2_var}][{dim3_var}];")  
                    
                    # Initialize values  
                    init_values = []  
                    if actual_dim == 3:  
                        for i in range(dim1):  
                            for j in range(dim2):  
                                for k in range(dim3):  
                                    value = input_val[i, j, k]  
                                    if input_val.dtype.kind in 'iu':  
                                        init_values.append(f"    {var_name}[{i}][{j}][{k}] = {int(value)};")  
                                    else:  
                                        init_values.append(f"    {var_name}[{i}][{j}][{k}] = {float(value)};")  
                    elif actual_dim < 3:  
                        # Handle lower dimensions by filling with zeros or stretching  
                        for i in range(dim1):  
                            for j in range(dim2):  
                                for k in range(dim3):  
                                    init_values.append(f"    {var_name}[{i}][{j}][{k}] = 0;")  
                    elif actual_dim > 3:  
                        # Flatten higher dimensions  
                        for i in range(dim1):  
                            for j in range(dim2):  
                                for k in range(dim3):  
                                    if all(idx < input_val.shape[dim] for dim, idx in enumerate([i, j, k])):  
                                        # Use first element of higher dimensions  
                                        value = input_val[(i, j, k) + (0,) * (actual_dim - 3)]  
                                        if input_val.dtype.kind in 'iu':  
                                            init_values.append(f"    {var_name}[{i}][{j}][{k}] = {int(value)};")  
                                        else:  
                                            init_values.append(f"    {var_name}[{i}][{j}][{k}] = {float(value)};")  
                                    else:  
                                        init_values.append(f"    {var_name}[{i}][{j}][{k}] = 0;")  
                      
                    array_initializations.append("\n".join(init_values))  

                  
                elif dim_to_use == 4:  # 4D array  
                    dim1 = shape[0] if actual_dim > 0 else 0  
                    dim2 = shape[1] if actual_dim > 1 else 0  
                    dim3 = shape[2] if actual_dim > 2 else 0  
                    dim4 = shape[3] if actual_dim > 3 else 0  
                    dim1_var = f"{var_name}_dim1"  
                    dim2_var = f"{var_name}_dim2"  
                    dim3_var = f"{var_name}_dim3"  
                    dim4_var = f"{var_name}_dim4"  
                      
                    var_declarations.append(f"    const int {dim1_var} = {dim1};")  
                    var_declarations.append(f"    const int {dim2_var} = {dim2};")  
                    var_declarations.append(f"    const int {dim3_var} = {dim3};")  
                    var_declarations.append(f"    const int {dim4_var} = {dim4};")  
                      
                    # Allocate 4D array  
                    array_allocations.append(f"    {base_type} {var_name}[{dim1_var}][{dim2_var}][{dim3_var}][{dim4_var}];")  
                      
                    # Initialize values  
                    init_values = []  
                    if actual_dim == 4:  
                        for i in range(dim1):  
                            for j in range(dim2):  
                                for k in range(dim3):  
                                    for l in range(dim4):  
                                        value = input_val[i, j, k, l]  
                                        if input_val.dtype.kind in 'iu':  
                                            init_values.append(f"    {var_name}[{i}][{j}][{k}][{l}] = {int(value)};")  
                                        else:  
                                            init_values.append(f"    {var_name}[{i}][{j}][{k}][{l}] = {float(value)};")  
                    elif actual_dim < 4:  
                        # Handle lower dimensions  
                        for i in range(dim1):  
                            for j in range(dim2):  
                                for k in range(dim3):  
                                    for l in range(dim4):  
                                        init_values.append(f"    {var_name}[{i}][{j}][{k}][{l}] = 0;")  
                      
                    array_initializations.append("\n".join(init_values))  
              
            # Add to kernel arguments  
            kernel_args.append(var_name)  

        # Create code to print output arrays    
        print_arrays = ""  
        
        # First, handle scalar outputs  
        for var_name, info in array_info.items():    
            if info['dimension'] == 0:  # Scalar output  
                print_arrays += f"""        
            // Print scalar output value       
            std::cout << "\\n{var_name} scalar value: " << *{var_name} << std::endl;        
        """  
        
        # Then handle array outputs  
        for var_name, info in array_info.items():    
            dimension = info['dimension']  
            if dimension == 0:  # Skip scalars (already handled)  
                continue  
                
            shape = info['shape']    
            # ... rest of your existing array printing code  
            if dimension == 1:  
                size = shape[0]  
                # Special handling for int8_t which can be printed as characters  
                if base_type.strip() == 'int8_t' or base_type.strip() == 'uint8_t':  
                    print_arrays += f"""        
                // Print {var_name} values (as integers to avoid character interpretation)       
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;  
                for (int i = 0; i < {size}; i++) {{        
                    // Cast to int to ensure numeric output  
                    std::cout << static_cast<int>({var_name}[i]);  
                    std::cout << " ";  // Use space as separator  
                }}  
                std::cout << std::endl << "END_DATA_{var_name}" << std::endl;       
            """  
                else:  
                    # Regular floating point or other integer types  
                    print_arrays += f"""        
                // Print {var_name} values        
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;        
                for (int i = 0; i < {size}; i++) {{        
                    std::cout << {var_name}[i] << " ";  
                }}  
                std::cout << std::endl << "END_DATA_{var_name}" << std::endl;        
            """  
            elif dimension == 2:  # 2D array  
                rows, cols = shape[0], shape[1]  
                
                # Special handling for int8_t/uint8_t types to avoid character interpretation  
                if base_type.strip() == 'int8_t' or base_type.strip() == 'uint8_t':  
                    print_arrays += f"""        
                // Print {var_name} values (as integers to avoid character interpretation)       
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;        
                for (int r = 0; r < {rows}; r++) {{        
                    for (int c = 0; c < {cols}; c++) {{        
                        // Cast to int to ensure numeric output  
                        std::cout << static_cast<int>({var_name}[r][c]) << " ";        
                    }}        
                    std::cout << std::endl;        
                }}  
                std::cout << "END_DATA_{var_name}" << std::endl;        
            """  
                else:  
                    # Regular handling for other types  
                    print_arrays += f"""        
                // Print {var_name} values        
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;        
                for (int r = 0; r < {rows}; r++) {{        
                    for (int c = 0; c < {cols}; c++) {{        
                        std::cout << {var_name}[r][c] << " ";        
                    }}        
                    std::cout << std::endl;        
                }}  
                std::cout << "END_DATA_{var_name}" << std::endl;        
            """  
            elif dimension == 3:  # 3D array  
                dim1, dim2, dim3 = shape[0], shape[1], shape[2]  
                
                # Special handling for int8_t/uint8_t  
                if base_type.strip() == 'int8_t' or base_type.strip() == 'uint8_t':  
                    print_arrays += f"""        
                // Print {var_name} values (as integers to avoid character interpretation)       
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;    
                for (int i = 0; i < {dim1}; i++) {{    
                    std::cout << "Slice " << i << ":" << std::endl;    
                    for (int j = 0; j < {dim2}; j++) {{        
                        for (int k = 0; k < {dim3}; k++) {{        
                            // Cast to int to ensure numeric output  
                            std::cout << static_cast<int>({var_name}[i][j][k]) << " ";        
                        }}        
                        std::cout << std::endl;        
                    }}    
                    std::cout << std::endl;    
                }}  
                std::cout << "END_DATA_{var_name}" << std::endl;        
            """  
                else:  
                    # Regular handling for other types  
                    print_arrays += f"""        
                // Print {var_name} values        
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;    
                for (int i = 0; i < {dim1}; i++) {{    
                    std::cout << "Slice " << i << ":" << std::endl;    
                    for (int j = 0; j < {dim2}; j++) {{        
                        for (int k = 0; k < {dim3}; k++) {{        
                            std::cout << {var_name}[i][j][k] << " ";        
                        }}        
                        std::cout << std::endl;        
                    }}    
                    std::cout << std::endl;    
                }}  
                std::cout << "END_DATA_{var_name}" << std::endl;        
            """  
            elif dimension == 4:  # 4D array  
                dim1, dim2, dim3, dim4 = shape[0], shape[1], shape[2], shape[3]  
                
                # Special handling for int8_t/uint8_t  
                if base_type.strip() == 'int8_t' or base_type.strip() == 'uint8_t':  
                    print_arrays += f"""        
                // Print {var_name} values (as integers to avoid character interpretation)       
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;    
                for (int i = 0; i < {dim1}; i++) {{    
                    std::cout << "Hyperslice " << i << ":" << std::endl;    
                    for (int j = 0; j < {dim2}; j++) {{    
                        std::cout << "Slice [" << i << "," << j << "]:" << std::endl;    
                        for (int k = 0; k < {dim3}; k++) {{        
                            for (int l = 0; l < {dim4}; l++) {{        
                                // Cast to int to ensure numeric output  
                                std::cout << static_cast<int>({var_name}[i][j][k][l]) << " ";        
                            }}        
                            std::cout << std::endl;        
                        }}    
                        std::cout << std::endl;    
                    }}    
                    std::cout << std::endl;    
                }}  
                std::cout << "END_DATA_{var_name}" << std::endl;        
            """  
                else:  
                    # Regular handling for other types  
                    print_arrays += f"""        
                // Print {var_name} values        
                std::cout << "\\n{var_name} values after kernel execution:" << std::endl;  
                std::cout << "START_DATA_{var_name}" << std::endl;    
                for (int i = 0; i < {dim1}; i++) {{    
                    std::cout << "Hyperslice " << i << ":" << std::endl;    
                    for (int j = 0; j < {dim2}; j++) {{    
                        std::cout << "Slice [" << i << "," << j << "]:" << std::endl;    
                        for (int k = 0; k < {dim3}; k++) {{        
                            for (int l = 0; l < {dim4}; l++) {{        
                                std::cout << {var_name}[i][j][k][l] << " ";        
                            }}        
                            std::cout << std::endl;        
                        }}    
                        std::cout << std::endl;    
                    }}    
                    std::cout << std::endl;    
                }}  
                std::cout << "END_DATA_{var_name}" << std::endl;        
            """  
          
        # Combine all code sections  
        declarations_str = "\n".join(var_declarations)  
        allocations_str = "\n".join(array_allocations)  
        initializations_str = "\n".join(array_initializations)  
        cleanup_str = "\n".join(cleanup_code)  
        kernel_call = f"    {kernel_name}({', '.join(kernel_args)});"  
          
        # Create the test C++ code  
        test_code = f"""      
#include <iostream>      
#include <vector>      
#include <string>      
#include <cstdlib>      
#include <cstring>      
#include <cstdint>  // Required for int8_t, uint8_t, etc.      
      
// Include the kernel code      
{kernel_code}      
      
// Main function to test the kernel      
int main() {{      
    std::cout << "Starting execution of {kernel_name}..." << std::endl;      
          
    // Variable declarations      
{declarations_str}      
          
    // Allocate memory for arrays      
{allocations_str}      
          
    // Initialize arrays      
{initializations_str}      
          
    // Call the kernel function      
    try {{      
{kernel_call}      
        std::cout << "Kernel executed successfully" << std::endl;      
              
        // Print array values for debugging      
{print_arrays}      
              
        // Cleanup      
{cleanup_str}      
              
        return 0;      
    }} catch (const std::exception& e) {{      
        std::cerr << "Error during kernel execution: " << e.what() << std::endl;      
              
        // Cleanup even on error      
{cleanup_str}      
              
        return 1;      
    }}      
}}      
"""  

        # Write the test code to a temporary file  
        temp_cpp = os.path.join(temp_dir, f"{kernel_name}_test.cpp")  
        with open(temp_cpp, 'w', encoding='utf-8') as f:  
            f.write(test_code)  
        with open("./temp.cc", 'w', encoding='utf-8') as f:  
            f.write(test_code)  
              
        # Compile the C++ file  
        exec_path = os.path.join(temp_dir, kernel_name)  
        vitis_path = os.getenv('XILINX_VITIS')
        if (vitis_path):
            vitis_include = f"{vitis_path}/include"
            compile_cmd = ["g++", "-std=c++11", temp_cpp, "-I", vitis_include, "-o", exec_path]  
        else:
            compile_cmd = ["g++", "-std=c++11", temp_cpp, "-o", exec_path]  

              
        try:  
            compile_result = subprocess.run(  
                compile_cmd,   
                capture_output=True,  
                # Don't use text=True  
                check=True  
            )  
            # Manually decode stdout and stderr  
            compile_stdout = compile_result.stdout.decode('utf-8', errors='replace')  
            compile_stderr = compile_result.stderr.decode('utf-8', errors='replace')  
        except subprocess.CalledProcessError as e:  
            # Decode error output  
            err_stdout = e.stdout.decode('utf-8', errors='replace') if e.stdout else ""  
            err_stderr = e.stderr.decode('utf-8', errors='replace') if e.stderr else ""  
            print(f"Compilation error: {err_stderr}")  
            return e.returncode, err_stdout, err_stderr, None  
              
        # Run the compiled program  
        try:  
            run_result = subprocess.run(  
                [exec_path],   
                capture_output=True,  
                # Don't specify text=True to get bytes instead of string  
                check=False  # Don't raise an exception on non-zero return code  
            )  
            
            # Manually decode stdout and stderr with error handling  
            stdout = run_result.stdout.decode('utf-8', errors='replace')  
            stderr = run_result.stderr.decode('utf-8', errors='replace')  
            # Extract result matrices if available
            result_matrices = []

            if output_matrix_indices is not None:
                # Ensure it's a list (in case a single int was passed)
                index_list = output_matrix_indices if isinstance(output_matrix_indices, (list, tuple)) else [output_matrix_indices]

                for idx in index_list:
                    matrix_name = f"input_{idx}"
                    if matrix_name in array_info:
                        info = array_info[matrix_name]
                        shape = info['shape']
                        dtype = info['dtype']
                        dimension = info['dimension']

                        # Extract and append the result matrix
                        result = extract_matrix_from_output(
                            stdout,
                            matrix_name,
                            dimension,
                            shape,
                            dtype
                        )
                        result_matrices.append(result)
                result_matrices = tuple(result_matrices)  # Convert to tuple for consistency
                
            return run_result.returncode, stdout, stderr, result_matrices  
  
        except Exception as e:  
            print(f"Error running the compiled program: {str(e)}")  
            return None, None, str(e), None  
                