# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

from CppHeaderParser import CppHeader
# from ml_dtypes import bfloat16
import logging
import re
import numpy as np
import json

logger = logging.getLogger("_PROMPT CONSTRUCTOR")

C_TO_NUMPY_DTYPE = {
    "int": "int",
    "int8_t": "int8",
    "ap_int<8>": "int8",
    "ap_int<16>": "int16",
    "ap_int<32>": "int32",
    "ap_int<64>": "int64",
    "ap_uint<8>": "uint8",
    "ap_uint<16>": "uint16",
    "ap_uint<32>": "uint32",
    "ap_uint<64>": "uint64",
    "uint8_t": "uint8",
    "int16_t": "int16",
    "uint16_t": "uint16",
    "int32_t": "int32",
    "uint32_t": "uint32",
    "int64_t": "int64",
    "uint64_t": "uint64",
    "bfloat16": "bfloat16",
    "float": "float32",
    "double": "float64",
}

# def extract_canonical(src):
#     match = re.search(r'void\s+\w+\s*\(.*?\)\s*{(.*)}', src, re.DOTALL)
#     if match:
#         return match.group(1).strip() + "\n}"
#     else:
#         raise Exception("Couldn't parse canonical solution.")

def parse_function_signature(signature):  
    """  
    Parse a C++ function signature to extract the function name and categorize parameters.  
      
    Args:  
        signature (str): A C++ function signature string  
          
    Returns:  
        dict: A dictionary containing:  
            - 'name': The function name  
            - 'buffers': List of parameter dictionaries that are pointers or arrays  
            - 'rtps': List of parameter dictionaries that are regular types (non-pointers)  
    """  
    # Extract function name  
    name_match = re.search(r'(?:\w+\s+)+(\w+)\s*\(', signature)  
    if not name_match:  
        raise ValueError("Could not extract function name from the signature")  
      
    function_name = name_match.group(1)  
      
    # Extract parameters section (everything between parentheses)  
    params_match = re.search(r'\((.*?)\)', signature)  
    if not params_match:  
        raise ValueError("Could not extract parameters from the signature")  
      
    params_str = params_match.group(1).strip()  
    if not params_str:  
        return {'name': function_name, 'parameters': [], 'buffers': [], 'rtps': []}  
      
    # Split parameters by comma, handling potential commas in template arguments  
    params = []  
    current_param = ""  
    bracket_level = 0  
      
    for char in params_str:  
        if char == ',' and bracket_level == 0:  
            params.append(current_param.strip())  
            current_param = ""  
        else:  
            current_param += char  
            if char == '<':  
                bracket_level += 1  
            elif char == '>':  
                bracket_level -= 1  
      
    if current_param.strip():  
        params.append(current_param.strip())  
      
    # Parse parameters  
    all_params = []  
    buffers = []  
    rtps = []  
      
    for param in params:  
        # Remove default value if any  
        param_parts = param.split('=')[0].strip()  
          
        # Extract pointer/array info  
        is_pointer = '*' in param_parts  
        is_array = '[]' in param_parts  
          
        # Normalize spacing around pointers for easier parsing  
        normalized_param = param_parts.replace('*', ' * ').replace('[', ' [').replace(']', '] ')  
          
        # Extract parameter name (last word before any array notation)  
        words = re.sub(r'\s+', ' ', normalized_param).strip().split()  
        param_name = ""  
          
        for i in range(len(words)-1, -1, -1):  
            if words[i] != '*' and not words[i].startswith('['):  
                param_name = words[i]  
                break  
          
        if not param_name:  
            continue  
          
        # Extract type (everything before the name, excluding pointer symbols)  
        type_end_idx = normalized_param.rfind(param_name)  
        if type_end_idx == -1:  
            continue  
              
        param_type = normalized_param[:type_end_idx].replace('*', '').strip()  
          
        # Create parameter dictionary similar to CppHeaderParser format  
        param_dict = {  
            "name": param_name,  
            "type": param_type,  
            "pointer": is_pointer,  
            "array": is_array,  
            "reference": '&' in param_parts  
        }  
          
        all_params.append(param_dict)  
          
        if is_pointer or is_array:  
            buffers.append(param_dict)  
          
        if not is_pointer:  
            rtps.append(param_dict)  
      
    return {  
        'name': function_name,  
        'parameters': all_params,  
        'buffers': buffers,  
        'rtps': rtps  
    } 
    
class PromptConstructor:
    def __init__(self, function_signature, description, behavioral, examples, tolerances=None): 
        #type is a list of types for the input buffers, has 2 parts: the dimension and the io direction. If it has 0 as Dimension, it is an RTP
        # --- Parse source + function ---
        # with open(source_path, "r") as f:
        #     source_code = f.read()
        # parsedcpp = CppHeader(function_signature, argType='string')

        # if len(parsedcpp.functions) > 1:
        #     raise RuntimeError("There should be only 1 function in the source code.")

        # func = parsedcpp.functions[0]
        # Parse the function signature  
        result = parse_function_signature(function_signature)  
          
        # Correctly assign the values from the dictionary  
        self.name = result['name']  
        self.parameters = result['parameters']  
        self.buffers = result['buffers']  
        self.rtps = result['rtps']
        self.examples = examples
        # self.buffers = [p for p in func['parameters'] if self._is_buffer(p)]
        # self.call = self._construct_call()

        # --- Extract canonical solution and signature---
        # with open(source_path, "r") as f:
        #     src = f.read()
        # canonical_solution = extract_canonical(src)
        # self.signature = self._construct_signature(src)

        # --- Generate test_vectors for each example --- 
        test_vectors_list = []

        for ex in self.examples:
            arrays = ex["inputs"]
            types = ex["types"]
            # --- Generate test_vectors for one example --- 
            test_vectors = {"inputs": [], "outputs": [], "rtps": []}
            output_indices = []
            input_arrays = []
            in_out_arrays = []
            rtp_values = []
            # Loop through saving the inputs and rtps and the index for the outputs in order to generate the test vectors
            for i, arr in enumerate(arrays):
                meta = types[i]
                in_out = meta["in_out"]
                array_dim = meta["array_dimension"]
                # Check if the array is an RTP or a buffer
                if in_out == "in":
                    if array_dim == 0:
                        # It's a scalar RTP
                        c_type = self.rtps[len(rtp_values)]["type"]
                        try:
                            np_type = C_TO_NUMPY_DTYPE[c_type]
                        except KeyError:
                            raise ValueError(f"Unsupported rtp type '{c_type}' in kernel '{self.name}'")
                        test_vectors["rtps"].append({
                            self.rtps[len(rtp_values)]["name"]: float(arr),
                            "dtype": np_type
                        })
                        rtp_values.append(arr)
                    # Check if the array is a buffer and input
                    else:
                        buf_name = self.buffers[len(input_arrays)]["name"]
                        test_vectors["inputs"].append({
                            buf_name: arr.tolist(),
                            "dtype": str(arr.dtype)
                        })
                        input_arrays.append(arr)
                elif in_out == "out":
                    # It's a buffer output
                    output_indices.append(i)  # Save the position; will generate later
                elif in_out == "inout": # Handle the case when it is both an input and an output
                    output_indices.append(i)  # Save the position; will generate later
                    if array_dim == 0:
                        # It's a scalar RTP
                        c_type = self.rtps[len(rtp_values)]["type"]
                        try:
                            np_type = C_TO_NUMPY_DTYPE[c_type]
                        except KeyError:
                            raise ValueError(f"Unsupported rtp type '{c_type}' in kernel '{self.name}'")
                        test_vectors["rtps"].append({
                            self.rtps[len(rtp_values)]["name"]: float(arr),
                            "dtype": np_type
                        })
                        rtp_values.append(arr)
                    # Check if the array is a buffer and input
                    else:
                        buf_name = self.buffers[len(input_arrays)]["name"]
                        test_vectors["inputs"].append({
                            buf_name: arr.tolist(),
                            "dtype": str(arr.dtype)
                        })
                        in_out_arrays.append(arr)

                else:
                    raise ValueError(f"Unknown in_out value '{in_out}' at index {i}")
                
            #Using the output indices, we can generate the outputs
            # logger.debug(input_arrays)
            args = [*input_arrays]

            if rtp_values:
                args.extend(rtp_values)

            if in_out_arrays:
                args.extend(in_out_arrays)

            outputs = behavioral(*args)
            # logger.debug(outputs)
            if not isinstance(outputs, tuple):
                outputs = (outputs,)
                
            for i, arr in enumerate(outputs):
                # Offset from input buffers
                buf_index = (len(input_arrays) + i) if (len(in_out_arrays) == 0) else (len(self.buffers) - len(in_out_arrays) + i) # The outputs are always last in the buffers so the indexes start after inputs. They overlap if they are inout so we need to take the length of inout_arrays from the end to start
                buf_name = self.buffers[buf_index]["name"]
                test_vectors["outputs"].append({
                    buf_name: arr.tolist(),
                    "dtype": str(arr.dtype)
                })
            # logger.debug(test_vectors["outputs"])
            test_vectors_list.append(test_vectors)

        # program_code = self._construct_wrapper()
        dataflow = self._construct_dataflow(test_vectors)
        
        # Some edge case kernels may fail to generate examples, we'll just skip for those
        try:
            examples = self._construct_examples(behavioral, test_vectors_list=test_vectors_list, num_examples=len(test_vectors_list), slice_size=8)
        except:
            logger.warning(f"The kernel {self.name} prompt won't have examples")
            examples = None
        
        if examples:
            prompt = f"""/*
{description}
{examples}
{dataflow}
*/
{function_signature} {{
"""
        else:
            prompt = f"""/*
{description}
{dataflow}
Use the following function signature to implement the kernel:
*/
{function_signature} {{
"""

        self.sample = {
            "kernel_name": self.name,
            "prompt": prompt,
            # "canonical_solution": canonical_solution, #FIXME: Find out what these are used for and if we need them
            # "program_code": program_code,
            # "test_vectors": test_vectors,
        }

        if tolerances:
            self.sample['tolerances'] = tolerances

    def write_json(self, filepath):
        logger.info(f"Writing JSON to {filepath}")
        with open(filepath, "w") as f:
            json.dump(self.sample, f)
            
    def _is_buffer(self, p):
        # Include pointer types or array declarations
        return p.get('pointer') or p.get('array', False) or '[]' in p.get('type', '')

    # Internal methods
    def _construct_signature(self, src):
        before_brace = src.split('{', 1)[0]
        return before_brace.strip()

    def _construct_call(self):
        buf = ", ".join([b['name'] for b in self.buffers])
        rtp = ", ".join([r['name'] for r in self.rtps])
        return f"{self.name}({buf}{', ' + rtp if rtp else ''})"

#     def _construct_wrapper(self):  #FIXME Do We need this?
#         buf = ", ".join([f"{b['type']}{b['name']}" for b in self.buffers])
#         rtp = ", ".join([f"{r['type']} {r['name']}" for r in self.rtps])
#         params = buf if not self.rtps else f"{buf}, {rtp}"
#         return f"""extern "C" {{
#     void {self.name}_wrapper({params}) {{
#         ::aie::set_rounding(aie::rounding_mode::positive_inf);
#         event0();
#         {self.call};
#         event1();
#     }}
# }}"""

    def _construct_dataflow(self, test_vectors):
        df = ""
        for d in test_vectors["inputs"]:
            for k, v in d.items():
                if k != "dtype":
                    df += f"{k} size: {np.size(v)}\n"
                    break
        for d in test_vectors["outputs"]:
            for k, v in d.items():
                if k != "dtype":
                    df += f"{k} size: {np.size(v)}\n"
                    break
        if "rtps" in test_vectors:
            for i, d in enumerate(test_vectors["rtps"]):
                for k, v in d.items():
                    if k != "dtype":
                        df += f"{k}: {v}\n"
        return f"This kernel should be optimized for the following input/output buffer shapes and parameters:\n{df.strip()}"

    # def _construct_examples(self, behavioral, input_arrays, rtp_values=None, num_examples=1, slice_size=8):
    #     examples = []
    #     arr0 = input_arrays[0]
    #     indices = [0, max(0, arr0.shape[0] - slice_size)] if arr0.ndim == 1 else [0]

    #     for idx in indices[:num_examples]:
    #         sliced = [a[idx:idx+slice_size] if a.ndim == 1 else a[:4, :4] for a in input_arrays]
    #         out = behavioral(*sliced, *rtp_values) if rtp_values else behavioral(*sliced)
    #         inputs_str = ', '.join(str(a.tolist()) for a in sliced)
    #         rtp_str = ', ' + ', '.join(map(str, rtp_values)) if rtp_values else ''
    #         examples.append(f">>> {self.name}({inputs_str}{rtp_str})\n{out.tolist()}")
    #     return "\n".join(examples)
    
    def _construct_examples(self, behavioral, test_vectors_list, num_examples=1, slice_size=8):
        examples = []

        for ex in test_vectors_list[:num_examples]:
            inputs = ex["inputs"]
            rtps = ex.get("rtps", [])

            # Extract just the arrays from the dicts
            input_arrays = [np.array(list(buf.values())[0]) for buf in inputs]
            rtp_values = [list(rtp.values())[0] for rtp in rtps] if rtps else []

            # Choose slicing indices
            arr0 = input_arrays[0]
            if arr0.ndim == 1:
                indices = [0, max(0, arr0.shape[0] - slice_size)]
            elif arr0.ndim in (2, 3):
                indices = [0]
    
            else:
                raise ValueError("Unsupported array dimension for example construction")

            for idx in indices[:1]:  # Just one slice per example
                sliced_inputs = []
                for a in input_arrays:
                    if a.ndim == 1:
                        sliced_inputs.append(a[idx:idx+slice_size])
                    elif a.ndim == 2:
                        sliced_inputs.append(a[:slice_size, :slice_size])
                    elif a.ndim == 3:
                        sliced_inputs.append(a[:slice_size, :slice_size, :slice_size])
                    else:
                        raise ValueError(f"Unsupported array dimension {a.ndim} for example construction")


                out = behavioral(*sliced_inputs, *rtp_values) if rtp_values else behavioral(*sliced_inputs)

                inputs_str = ', '.join(str(a.tolist()) for a in sliced_inputs)
                rtp_str = ', ' + ', '.join(map(str, rtp_values)) if rtp_values else ''
                
                # Handle tuple output safely
                if isinstance(out, tuple):
                    out_str = '\n'.join(str(o.tolist() if isinstance(o, np.ndarray) else o) for o in out)
                else:
                    out_str = out.tolist() if isinstance(out, np.ndarray) else str(out)

                examples.append(f">>> {self.name}({inputs_str}{rtp_str})\n{out_str}")
        return "\n".join(examples)
