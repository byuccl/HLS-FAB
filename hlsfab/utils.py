# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from CppHeaderParser import CppHeader
import numpy as np
import subprocess
import json
import re
import os
import sys
import traceback  
import importlib.util
import logging
from ml_dtypes import bfloat16

logger = logging.getLogger("UTILS")

def generate_inputs(inputs_info, seed=0):
    """
    generate the inputs that will be used in the kernel function.
    """
    np.random.seed(seed)
    if seed == -1:
        # Load from a saved npz file if seed is -1
        data = np.load('data.npz')
        inputs = [data[x['name']] for x in inputs_info]

    else: 
        # Generate random inputs based on the provided inputs_info
        inputs = []
        for info in inputs_info:
            shape = info.get('shape', (1,))
            dtype = info['dtype']
            min_val = info.get('min', 0)
            max_val = info.get('max', 127)
            if info['iotype'] == "in" or info['iotype'] == "inout": #if an input or any kind it needs some data
                if info['array_dimension'] > 0: 
                    # Generate a random array of the specified shape and dtype
                    if "float" in dtype:
                        # Generate a float array
                        inputs.append(np.random.uniform(min_val, max_val, size=shape).astype(dtype).flatten())
                    elif "int" in dtype:
                        inputs.append(np.random.randint(min_val, max_val, size=shape, dtype=dtype).flatten())

                else:
                    # Generate a scalar value
                    if "float" in info['dtype']:
                        inputs.append(np.random.uniform(min_val, max_val, 1).astype(dtype).item())
                    elif "int" in info['dtype']:
                        inputs.append(np.random.randint(min_val, max_val, dtype=dtype))
            elif info['iotype'] == "out": # if its an output, we just need to create a zero array of the correct shape
                output = np.zeros(shape, dtype=dtype)
                if info['array_dimension'] > 0:
                    # Flatten the output array to match the expected shape
                    inputs.append(output.flatten())
                else:
                    # For scalar outputs, just append the zero value
                    inputs.append(output.item())


    return { "inputs": inputs, "types": inputs_info } 

#only used in csim.py
def import_module_members(kernel_name, member_names=None):      
    """      
    Dynamically import specified functions from the kernel's generate.py file.      
    """      
    if member_names is None:      
        member_names = ["behavioral"]      
          
    # Get the project root directory (1 level up from this file)      
    project_root = os.path.dirname(os.path.dirname(__file__))     
     
          
    # Add project root to sys.path to allow imports to work      
    if project_root not in sys.path:      
        sys.path.insert(0, project_root)      
          
    # Construct the relative path to the generate.py file      
    module_path = os.path.join(project_root, "dataset", "kernels", kernel_name, "generate.py")    
          
    if not os.path.exists(module_path):      
        logger.warning(f"Could not find generate.py for kernel '{kernel_name}' at path: {module_path}")      
        return None      
          
    # Create a unique module name to avoid conflicts      
    module_name = f"dynamic_kernel_{kernel_name}"      
          
    # Create spec from file path      
    spec = importlib.util.spec_from_file_location(module_name, module_path)      
    if spec is None:      
        logger.error(f"Could not create module spec for {module_path}")      
        return None      
          
    # Create module from spec      
    module = importlib.util.module_from_spec(spec)      
          
    # Add the module to sys.modules      
    sys.modules[module_name] = module      
          
    # Execute the module      
    try:      
        spec.loader.exec_module(module)      
    except Exception as e:      
        logger.error(f"Error loading module {module_path}: {e}")  
        logger.error("Detailed traceback:")  
        traceback.print_exc()  
        return None      
          
    # Create a dictionary to hold the requested functions      
    members = {}      
          
    # Get the requested functions      
    for mem_name in member_names:      
        if hasattr(module, mem_name):      
            members[mem_name] = getattr(module, mem_name)      
        else:      
            logger.warning(f"Module {module_path} does not have a {mem_name} member")      

    return members
def get_ap_cpp_type_str(dtype_str):
    dtype_map = {
        "int8": "ap_int<8>", "uint8": "ap_uint<8>",
        "int16": "ap_int<16>", "uint16": "ap_uint<16>",
        "int32": "ap_int<32>", "uint32": "ap_uint<32>",
        "int64": "ap_int<64>", "uint64": "ap_uint<64>",
        "float16": "ap_float<16,5>",
        "bfloat16": "ap_float<16,8>",
        "float32": "float", "float64": "double",
        "bool": "bool",
        "int": "int",
        "float": "float",
    }
    return dtype_map.get(dtype_str.lower(), "void*")
def get_std_cpp_type_str(dtype_str):
    dtype_map = {
        "int8": "int8_t", "uint8": "uint8_t",
        "int16": "int16_t", "uint16": "uint16_t",
        "int32": "int32_t", "uint32": "uint32_t",
        "int64": "int64_t", "uint64": "uint64_t",
        "float16": "half_float_type",
        "float32": "float", "float64": "double",
        "bool": "bool",
        "int": "int",
        "float": "float",
    }
    return dtype_map.get(dtype_str.lower(), "void*")

#used in ever generate
def generate_examples(inputs_info):
    example1 = generate_inputs(inputs_info=inputs_info, seed=0)
    example2 = generate_inputs(inputs_info=inputs_info, seed=1)
    # Combine the inputs and types into a single list
    examples = [example1, example2]
    return examples
def generate_types(inputs):  
    """
    Generate comprehensive type information for a list of inputs/outputs,
    including variable name, array dimension, I/O direction, and data type.

    Args:
        inputs (list): A list of input and output arrays/values.
                       These should be the actual variables passed to the kernel
                       from the `generate_inputs` function.

    Returns:
        list: A list of dictionaries, where each dictionary represents
              the type information for a corresponding element in the `inputs` list.
              Each dictionary will have:
              - 'name' (str): The name of the variable in the caller's scope.
              - 'array_dimension' (int): The number of dimensions for arrays (0 for scalars).
              - 'in_out' (str): "in", "out", or "inout" based on variable name convention.
              - 'dtype' (str): The data type (e.g., "int32", "float64", "int", "float", "bool").
    """
    types_info = []
    
    import inspect  
    # Get the calling frame to inspect variable names
    # This assumes generate_types is called directly by the generate_inputs function
    frame = inspect.currentframe().f_back

    # Create a mapping from object identity to variable name
    # This is more robust than iterating through f_locals directly,
    # especially if multiple variables have the same value.
    var_name_map = {}
    for name, value in frame.f_locals.items():
        # Using id() for identity check to correctly map original variable names
        # to the objects in the 'inputs' list.
        # Check if the value is one of the inputs to avoid matching non-input variables.
        if id(value) in [id(item) for item in inputs]:
            var_name_map[id(value)] = name
      
    for i, item in enumerate(inputs):
        type_dict = {}

        # 1. Get Variable Name
        # Try to retrieve the original variable name from the caller's scope
        # Fallback to a generic name if not found
        var_name = var_name_map.get(id(item), f"arg_{i}")
        type_dict["name"] = var_name

        # 2. Set Array Dimension
        if isinstance(item, np.ndarray):
            type_dict["array_dimension"] = item.ndim
            type_dict["dtype"] = str(item.dtype)
            type_dict["shape"] = list(item.shape) # Store actual shape as a list
            type_dict["total_elements"] = item.size # Store total elements
        elif isinstance(item, np.integer):
            type_dict["array_dimension"] = 0
            type_dict["dtype"] = str(item.dtype) # e.g., 'int32', 'int64'
            type_dict["shape"] = [] # Scalar has no shape
            type_dict["total_elements"] = 1 # Scalar has 1 element
        elif isinstance(item, np.floating):
            type_dict["array_dimension"] = 0
            type_dict["dtype"] = str(item.dtype) # e.g., 'float32', 'float64'
            type_dict["shape"] = [] # Scalar has no shape
            type_dict["total_elements"] = 1 # Scalar has 1 element
        elif isinstance(item, np.bool_):
            type_dict["array_dimension"] = 0
            type_dict["dtype"] = str(item.dtype) # e.g., 'bool_'
            type_dict["shape"] = [] # Scalar has no shape
            type_dict["total_elements"] = 1 # Scalar has 1 element
        else:
            type_dict["array_dimension"] = 0
            # Get Python scalar type
            if isinstance(item, int): type_dict["dtype"] = "int"
            elif isinstance(item, float): type_dict["dtype"] = "float"
            elif isinstance(item, bool): type_dict["dtype"] = "bool"
            else: type_dict["dtype"] = "unknown" # Fallback for other scalar types
            type_dict["shape"] = [] # Scalar has no shape
            type_dict["total_elements"] = 1 # Scalar has 1 element

        # 3. Set in_out based on variable name convention
        # Convert the determined variable name to lowercase for consistent checking
        lower_var_name = var_name.lower()
        if "inout" in lower_var_name:
            type_dict["in_out"] = "inout"
        elif "output" in lower_var_name or "out" in lower_var_name:
            type_dict["in_out"] = "out"
        elif "input" in lower_var_name or "in" in lower_var_name:
            type_dict["in_out"] = "in"
        else:
            # Default to "in" if no clear convention is found in the name
            # You might want to refine this default based on your common patterns
            type_dict["in_out"] = "in"

        types_info.append(type_dict)

    return types_info



def write_table_header(logfile):
    """Write the header of the ASCII table to the logfile."""
    # Define headers and column widths
    headers = ["Kernel", "LLM Time", "Dynamatic time", "Vitis CSIM time", "Vitis CSYNTH time"]
    col_widths = [45, 20, 20, 20, 20]
    
    # Create top border
    border = "+"
    for width in col_widths:
        border += "-" * width + "+"
    logfile.write(border + "\n")
    
    # Write header row
    header_row = "|"
    for header, width in zip(headers, col_widths):
        header_row += header.center(width) + "|"
    logfile.write(header_row + "\n")
    
    # Write separator after headers
    logfile.write(border + "\n")
    logfile.flush()
def add_table_row(logfile, kernel, attempt=1, llm_time = "-", dynamatic_time = "-", vitis_csim_time = "-", vitis_csynth_time = "-"):
    """Add a row to the ASCII table in the logfile."""
    col_widths = [45, 20, 20, 20, 20]
    
    # Prepare the data
    kernel_attempt = f"{kernel}_attempt_{attempt+1}"
    data = [kernel_attempt, llm_time, dynamatic_time, vitis_csim_time, vitis_csynth_time]
    
    # Create the data row
    data_row = "|"
    for value, width in zip(data, col_widths):
        # Convert to string and center in the column
        value_str = str(value) if value is not None else ""
        data_row += value_str.center(width) + "|"
    
    logfile.write(data_row + "\n")
    logfile.flush()
def write_table_footer(logfile):
    """Write the footer (bottom border) of the ASCII table to the logfile."""
    col_widths = [45, 20, 20, 20, 20]
    
    # Create bottom border
    border = "+"
    for width in col_widths:
        border += "-" * width + "+"
    
    logfile.write(border + "\n")
    logfile.flush()


