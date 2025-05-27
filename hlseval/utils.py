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

def get_kernel_code(test: dict, solutions_path: str = None) -> str:
    """Fetch the kernel code from the provided solution path, if none provided default 
    to canonical solution."""
    if not solutions_path:
        return test['prompt'] + test['canonical_solution']
    
    with open(os.path.join(solutions_path, f"{test['kernel_name']}.json"), 'r') as sol_file:
        solution = json.load(sol_file)
        if not solution.get('code'):
            print(f"No code available in {solutions_path} for {test['kernel_name']}")
            return None

        srccode = solution['code']
        
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
        print(f"Trace written to {output_name}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Trace failed\n{e.output}")
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
        print(f"Warning: Could not find generate.py for kernel '{kernel_name}' at path: {module_path}")      
        return None      
          
    # Create a unique module name to avoid conflicts      
    module_name = f"dynamic_kernel_{kernel_name}"      
          
    # Create spec from file path      
    spec = importlib.util.spec_from_file_location(module_name, module_path)      
    if spec is None:      
        print(f"Error: Could not create module spec for {module_path}")      
        return None      
          
    # Create module from spec      
    module = importlib.util.module_from_spec(spec)      
          
    # Add the module to sys.modules      
    sys.modules[module_name] = module      
          
    # Execute the module      
    try:      
        spec.loader.exec_module(module)      
    except Exception as e:      
        print(f"Error loading module {module_path}: {e}")  
        print("Detailed traceback:")  
        traceback.print_exc()  
        return None      
          
    # Create a dictionary to hold the requested functions      
    functions = {}      
          
    # Get the requested functions      
    for func_name in function_names:      
        if hasattr(module, func_name):      
            functions[func_name] = getattr(module, func_name)      
        else:      
            print(f"Warning: Module {module_path} does not have a {func_name} function")      
          
    return functions    


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
    if isinstance(input_val, bool):  
        return "bool"  
    elif isinstance(input_val, int):  
        return "int"  
    elif isinstance(input_val, np.int8):  
        return "int8_t"  
    elif isinstance(input_val, np.int16):  
        return "int16_t"    
    elif isinstance(input_val, np.int32):  
        return "int32_t"   
    elif isinstance(input_val, np.int64):  
        return "int64_t"  
    elif isinstance(input_val, np.float32):  
        return "float"  
    elif isinstance(input_val, float):
        return "double"
    elif isinstance(input_val, np.float64):
        return "double"
    elif isinstance(input_val, np.ndarray):  
        # Map numpy dtypes to C++ types  
        dtype = input_val.dtype  
        if dtype == np.float32:  
            base_type = "float"  
        elif dtype == np.float64:  
            base_type = "double"  
        elif dtype == np.int8:  
            base_type = "int8_t"  
        elif dtype == np.int16:  
            base_type = "int16_t"  
        elif dtype == np.int32:  
            base_type = "int32_t"  
        elif dtype == np.int64:  
            base_type = "int64_t"  
        elif dtype == np.uint8:  
            base_type = "uint8_t"  
        elif dtype == np.uint16:  
            base_type = "uint16_t"  
        elif dtype == np.uint32:  
            base_type = "uint32_t"  
        elif dtype == np.uint64:  
            base_type = "uint64_t"  
        elif dtype == np.bool_:  
            base_type = "bool"  
        else:  
            raise ValueError(f"Unsupported numpy dtype: {dtype}")  
            
        # Add appropriate number of * based on dimensions  
        dimension = type_info.get('dimension', len(input_val.shape)) if type_info else len(input_val.shape)  
        if dimension == 0:  
            return base_type  
        return base_type + '*' * dimension  
    else:  
        return "void*"  # Default for unknown types  
    
def generate_types(inputs):  
    """  
    Generate types information for a list of inputs/outputs.  
      
    Args:  
        inputs: List of input and output arrays/values  
          
    Returns:  
        List of dictionaries with array_dimension and in_out information  
    """  
    types = []  
      
    # Get the calling frame to inspect variable names  
    import inspect  
    frame = inspect.currentframe().f_back  
      
    # Find all variables in the caller's scope that match items in inputs  
    input_vars = {}  
    for name, value in frame.f_locals.items():  
        for i, item in enumerate(inputs):  
            if item is value:  # Using 'is' to check identity  
                input_vars[i] = name  
      
    for i in range(len(inputs)):  
        types.append({})  
          
        # Set array dimension  
        if isinstance(inputs[i], np.ndarray):  
            types[i]["array_dimension"] = inputs[i].ndim  
        else:  
            types[i]["array_dimension"] = 0  
          
        # Set in_out based on variable name  
        if i in input_vars:  
            var_name = input_vars[i].lower()  
            if "input" in var_name:  
                types[i]["in_out"] = "in"  
            elif "output" in var_name:  
                types[i]["in_out"] = "out"  
            else:  
                # Default or error handling  
                types[i]["in_out"] = "unknown"  
        else:  
            types[i]["in_out"] = "unknown"  
              
    return types  