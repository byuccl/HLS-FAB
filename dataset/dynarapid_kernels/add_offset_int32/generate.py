import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void add_offset_int32(ap_int<32>* in_buffer, ap_int<32>* out_buffer, ap_int<32> offset)"

description = f"""
This HLS kernel adds a scalar 32-bit integer offset to every element of the input vector of 32-bit integers.
Parameters to the function include:
    - A pointer to the input buffer of {num_values} ap_int<32> values.
    - A pointer to the output buffer of {num_values} ap_int<32> values.
    - An ap_int<32> scalar offset value.
The kernel computes the sum of each element in the input buffer with the offset and stores the result in the output buffer.
The output values are clipped to the range of -1232 to 127 to fit within the 32-bit integer range.
"""

inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int32",
        "iotype": "in",
        "min": -100,
        "max": 100,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "offset",
        "array_dimension": 0,
        "shape": (),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "in",
        "min": -40,
        "max": 40,
    }
]

def behavioral(in_buffer, offset):
    res = in_buffer.astype(np.int16) + int(offset)
    res = np.clip(res, -2**31, 2**31-1)
    return res.astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")