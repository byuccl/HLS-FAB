import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void sqrt_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

short_description = "A kernel that computes the elementwise square root of a bfloat16 vector. Negative values are clamped to 0.0 in the output."

description = f"""
This HLS kernel computes the elementwise square root of a vector of bfloat16 values.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
Negative input values are clamped to 0.0 in the output.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -9,  # Minimum value for the input buffer
        "max": 9.0,  # Maximum value for the input buffer
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
    return np.sqrt(np.clip(input_buffer.astype(np.float32), 0.0, None)).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)


pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")