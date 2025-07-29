import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void max_abs_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

short_description = f"This HLS kernel finds the maximum absolute value in a bfloat16 input vector of size {num_values} and writes the result as a scalar bfloat16 in the output buffer."
description = f"""
This HLS kernel finds the maximum absolute value in a bfloat16 input vector of size {num_values} and writes the result as a scalar bfloat16 in the output buffer.
    First parameter: Pointer to the input buffer of {num_values} ap_float<16,8> values.
    Second parameter: Pointer to the output buffer where the maximum absolute value will be stored as a single ap_float<16,8> value.
The kernel iterates over the input buffer, computes the absolute value of each element, and keeps track of the maximum absolute value found.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -7,  # Minimum value for the input buffer
        "max": 7.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 0,  # Number of dimensions in the output buffer
        "shape": (),  # Shape of the output buffer
        "total_elements": 1,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    absmax = np.max(np.abs(input_buffer.astype(np.float32)))
    return bfloat16(absmax)

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")