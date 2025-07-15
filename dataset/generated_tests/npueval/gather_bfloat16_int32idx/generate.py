import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
src_buffer_size = 256

function_signature = "void gather_bfloat16_int32idx(ap_float<16,8>* input_data, ap_int<32>* indices, ap_float<16,8>* output_data)"

description = f"""
This HLS kernel implements a gather operation using bfloat16 data type for input and output buffers, and int32 for indices.
The kernel takes three parameters:
    1. A pointer to the input buffer containing {src_buffer_size} elements of type ap_float<16,8>.
    2. A pointer to the indices buffer containing {src_buffer_size} elements of type ap_int<32>.
    3. A pointer to the output buffer where the gathered elements will be stored, also containing {src_buffer_size} elements of type ap_float<16,8>.
For each index in the indices buffer, the kernel retrieves the corresponding element from the input buffer and stores it in the output buffer.
"""

inputs_info = [
    {
        "name": "input_data",
        "array_dimension": 1,
        "shape": (src_buffer_size,),
        "total_elements": src_buffer_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "indices",
        "array_dimension": 1,
        "shape": (src_buffer_size,),
        "total_elements": src_buffer_size,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": src_buffer_size - 1,
    },
    {
        "name": "output_data",
        "array_dimension": 1,
        "shape": (src_buffer_size,),
        "total_elements": src_buffer_size,
        "dtype": "bfloat16",
        "iotype": "out",
    }
]

def behavioral(input_data, indices):
    return input_data[indices]

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")