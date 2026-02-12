import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void rsqrt_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

short_description = "This HLS kernel computes the reciprocal square root (1/sqrt(x)) for each element of the bfloat16 input vector of length 256, producing a bfloat16 output vector. Nonpositive inputs yield 0 by convention."

description = f"""
This HLS kernel computes the reciprocal square root (1/sqrt(x)) for each element of the bfloat16 input vector of length {num_values}, producing a bfloat16 output vector.
The kernel takes two parameters:
    1. A pointer to the input buffer of {num_values} ap_float<16,8> values.
    2. A pointer to the output buffer of {num_values} ap_float<16,8> values.
For nonpositive inputs, the output is set to 0 by convention.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0.1,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    result = np.zeros_like(input_buffer)
    mask = input_buffer > 0
    result[mask] = 1.0 / np.sqrt(input_buffer[mask])
    return result.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)


pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")