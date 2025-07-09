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
from .hlscoder import HLSCoder

logger = logging.getLogger("__main__")


def import_module_functions(kernel_name, function_names=None):      
    """      
    Dynamically import specified functions from the kernel's generate.py file.      
    """      
    if function_names is None:      
        function_names = ["generate_inputs", "behavioral"]      
          
    # Get the project root directory (2 levels up from results/testing)      
    project_root = os.path.dirname(os.path.dirname(__file__))     
     
          
    # Add project root to sys.path to allow imports to work      
    if project_root not in sys.path:      
        sys.path.insert(0, project_root)      
          
    # Construct the relative path to the generate.py file      
    module_path = os.path.join(project_root, "dataset", "kernels", kernel_name, "generate.py")    
          
    if not os.path.exists(module_path):      
        logger.warning(f"UTILS: Could not find generate.py for kernel '{kernel_name}' at path: {module_path}")      
        return None      
          
    # Create a unique module name to avoid conflicts      
    module_name = f"dynamic_kernel_{kernel_name}"      
          
    # Create spec from file path      
    spec = importlib.util.spec_from_file_location(module_name, module_path)      
    if spec is None:      
        logger.error(f"UTILS: Could not create module spec for {module_path}")      
        return None      
          
    # Create module from spec      
    module = importlib.util.module_from_spec(spec)      
          
    # Add the module to sys.modules      
    sys.modules[module_name] = module      
          
    # Execute the module      
    try:      
        spec.loader.exec_module(module)      
    except Exception as e:      
        logger.error(f"UTILS: Error loading module {module_path}: {e}")  
        logger.error("UTILS: Detailed traceback:")  
        traceback.print_exc()  
        return None      
          
    # Create a dictionary to hold the requested functions      
    functions = {}      
          
    # Get the requested functions      
    for func_name in function_names:      
        if hasattr(module, func_name):      
            functions[func_name] = getattr(module, func_name)      
        else:      
            logger.warning(f"UTILS: Module {module_path} does not have a {func_name} function")      
          
    return functions    
 
def get_cpp_type_str(dtype_str):
    dtype_map = {
        "int8": "ap_int<8>", "uint8": "ap_uint<8>",
        "int16": "ap_int<16>", "uint16": "ap_uint<16>",
        "int32": "ap_int<32>", "uint32": "ap_uint<32>",
        "int64": "ap_int<64>", "uint64": "ap_uint<64>",
        "float16": "half_float_type",
        "float32": "float", "float64": "double",
        "bool": "bool",
        "int": "int",
        "float": "float",
    }
    return dtype_map.get(dtype_str.lower(), "void*")


def generate_examples(generate_function):
    # Example 1
    example1 = generate_function()
    example1_inputs = example1["inputs"]
    example1_types = example1["types"]

    # Example 2
    example2 = generate_function(seed=1)
    example2_inputs = example2["inputs"]
    example2_types = example2["types"]

    # Combine the inputs and types into a single list
    examples = [example1, example2]
    return examples

def get_kernel_code(hls_flow, test: dict, solutions_path: str = None) -> str:
    """Fetch the kernel code from the provided solution path, if none provided default 
    to canonical solution."""
    if not solutions_path:
        return test['prompt'] + test['canonical_solution']
    
    with open(os.path.join(solutions_path, f"{test['kernel_name']}.json"), 'r') as sol_file:
        solution = json.load(sol_file)
        if not solution.get('code'):
            if not solution.get('content'):
                logger.warning(f"UTILS: No code available in {solutions_path} for {test['kernel_name']}")
                return None
            srccode = solution['content']
            return srccode


        srccode = solution['code']
        if hls_flow == "vitis":
            # if gpt decides to be too helpful and adds a main()... remove it
            srccode = re.sub(r'int\s+main\s*\([^)]*\)\s*{[^{}]*({[^{}]*}[^{}]*)*}', '', srccode, flags=re.DOTALL)
            
            # cppheaderparser will complain if we don't remove trailing comments
            srccode = srccode.split('// extern "C"')[0]
            
        return srccode

def extract_buffers(test):
    """Specific helper for the AIEval dataset - parses the test dictionary and returns
    input buffers, output buffers and RTPs as separate lists.
    """
    input_buffers = []
    for x in test['test_vectors']['inputs']:
        array, dtype = list(x.values())
        input_buffers.append(np.array(array, dtype=dtype))
    
    output_buffers = []
    for x in test['test_vectors']['outputs']:
        array, dtype = list(x.values())
        output_buffers.append(np.array(array, dtype=dtype))

    rtps = []
    if test['test_vectors'].get('rtps') != None:
        for rtp in test['test_vectors']['rtps']:
            array, dtype = rtp.values()
            rtps.append(np.array(array, dtype=dtype))
            # rtp_names.append(list(rtp.keys())[0])
    
    return input_buffers, output_buffers, rtps

def trace_to_json(trace_file: str, mlir_file: str, output_name: str="trace.json"):
    """Subprocesses wrapper over parse_trace.py utility.

    Parameters
    ----------
    trace_file : str
        The .txt trace file of 32-byte codes.
    mlir_file : str
        Path to the corresponding MLIR file for the design being traced.
    output_name : str, optional
        Path to output json file. You can analyze it using tools like https://ui.perfetto.dev
    """
    command = [
        os.environ["MLIR_AIE_INSTALL_DIR"]+"/../../programming_examples/utils/parse_trace.py",
        "--filename",
        trace_file,
        "--mlir",
        mlir_file,
        "--colshift",
        "1"
    ]

    try:
        result = subprocess.check_output(command, stderr=subprocess.STDOUT, text=True)
        with open(output_name, 'w') as f:
            f.write(result)
        logger.info(f"UTILS: Trace written to {output_name}")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"UTILS: Trace failed\n{e.output}")
        return e.output

def get_cycles(trace_path):
    """This helper function should only be used to extract cycle counts
    from NPUEval trace files where the expectation is to have exactly 1 of
    each event0 and event1.
    """
    with open(trace_path, 'r') as f:
        data = json.load(f)

    event0 = []
    event1 = []
    try:
        for x in data:
            if (x['name'] == "INSTR_EVENT_0") and (x['ph'] == 'B'):
                event0.append(x['ts'])
        
            if x['name'] == "INSTR_EVENT_1" and x['ph'] == 'B':
                event1.append(x['ts'])
        
        return event1[0]-event0[0]
    except:
        return np.inf

def get_vector_time(trace):
    """This function extracts the total time spent on the vectorized unit
    from an NPUEval AIE trace (this must have exactly 1 event0 and 1 event1
    sandwiching the kernel call).
    """
    with open(trace, 'r') as f:
        data = json.load(f)

    start, end = None, None
    
    # find start and end
    for x in data:
        if (x['name'] == "INSTR_EVENT_0") and (x['ph'] == 'B'):
            start = x['ts']
        if x['name'] == "INSTR_EVENT_1" and x['ph'] == 'B':
            end = x['ts']

    if not start or not end:
        return 0

    total_duration = 0
    stack = []
    
    for event in data:
        if event['name'] == "INSTR_VECTOR":
            if event['ts'] < start:
                continue
            
            if event['ts'] > end:
                continue
            
            if event['ph'] == 'B':
                stack.append(event)
            elif event['ph'] == 'E' and stack:
                # Get matching begin event
                begin_event = stack.pop()
                # Calculate duration for this pair
                duration = event['ts'] - begin_event['ts']
                total_duration += duration
            
    return total_duration/(end-start)


# Helper function to determine C++ type for an input  
def get_cpp_type(input_val, type_info=None):  
    """Determine the appropriate C++ type for an input value
        input_value: variable that we want to find the type.
        type_info: a dict with information about type.
    """  

    # Use type_info if provided and it has a 'type' field  
    if type_info and 'type' in type_info:  
        return type_info['type']  
        
    # Otherwise infer from the input value  
    if isinstance(input_val, bool):  return "bool"  
    elif isinstance(input_val, int): return "int"  
    elif isinstance(input_val, np.int8): return "int8_t"  
    elif isinstance(input_val, np.int16): return "int16_t"    
    elif isinstance(input_val, np.int32): return "int32_t"   
    elif isinstance(input_val, np.int64): return "int64_t"  
    elif isinstance(input_val, np.uint8): return "uint8_t"  
    elif isinstance(input_val, np.uint16): return "uint16_t"    
    elif isinstance(input_val, np.uint32): return "uint32_t"   
    elif isinstance(input_val, np.uint64): return "uint64_t"  
    elif isinstance(input_val, np.float32): return "float"  
    elif isinstance(input_val, float): return "double"
    elif isinstance(input_val, np.float64): return "double"
    elif isinstance(input_val, np.ndarray):  
        # Map numpy dtypes to C++ types  
        dtype = input_val.dtype  
        if dtype == np.float32: base_type = "float"
        elif dtype == np.float64: base_type = "double"  
        elif dtype == np.int8: base_type = "int8_t"  
        elif dtype == np.int16: base_type = "int16_t"  
        elif dtype == np.int32: base_type = "int32_t"  
        elif dtype == np.int64: base_type = "int64_t"  
        elif dtype == np.uint8: base_type = "uint8_t"  
        elif dtype == np.uint16: base_type = "uint16_t"  
        elif dtype == np.uint32: base_type = "uint32_t"  
        elif dtype == np.uint64: base_type = "uint64_t"  
        elif dtype == np.bool_: base_type = "bool"  
        else: raise ValueError(f"Unsupported numpy dtype: {dtype}")  
        # Add appropriate number of * based on dimensions  
        dimension = type_info.get('dimension', len(input_val.shape)) if type_info else len(input_val.shape)  
        if dimension == 0:  
            return base_type  
        return base_type + '*' * dimension  
    else:  
        return "void*"  # Default for unknown types  

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
        elif "output" in lower_var_name:
            type_dict["in_out"] = "out"
        elif "input" in lower_var_name:
            type_dict["in_out"] = "in"
        else:
            # Default to "in" if no clear convention is found in the name
            # You might want to refine this default based on your common patterns
            type_dict["in_out"] = "in"

        types_info.append(type_dict)

    return types_info

def generate_one_completion(prompt, url, workflow, hls_flow, system_prompt, model="gpt-4o", temperature=0.0, top_p=1.0, attempts=1, rag=True):
    
    coder = HLSCoder(
        model=model,
        workflow=workflow,
        hls_flow=hls_flow,
        system_prompt=system_prompt,
        temperature=temperature,
        top_p=top_p,
        url=url,
        attempts=attempts,
        rag=rag
    )
    response = coder(prompt)

    result = {
        "code": coder.extract_codeblock(response["response"]),
        "stats": {
            "token_usage": response["token_usage"],
            "history": response["history"],
        },
    }

    return result
