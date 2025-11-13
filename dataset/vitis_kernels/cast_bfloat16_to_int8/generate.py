import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void cast_bfloat16_to_int8(ap_float<16,8> *input_buffer, ap_int<8> *output_buffer)"

description = f"""
This HLS kernel casts each element of a bfloat16 vector input to int8, performing rounding to the nearest integer and saturating to the int8 value range [-128, 127].
Parameters to the function include two pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {num_values}-element array of ap_int<8> values to be modified and viewed as the output to the function.
The kernel processes each element of the input buffer, converts it to a float, rounds it to the nearest integer, and saturates the result to fit within the int8 range.
"""

inputs_info = [
    {
        "name": "input_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -70,  # Minimum value for the input buffer
        "max": 70,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "int8",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    x = input_buffer.astype(np.float32)
    r = np.trunc(x + np.sign(x) * 0.5)
    r = np.clip(r, -128, 127)
    return r.astype(np.int8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")