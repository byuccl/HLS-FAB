import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

vector_size = 256  # Length of the input vector

function_signature = "void ceil_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel implements a ceil operation on a vector of bfloat16 numbers, rounding each element up to the nearest integer.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be used as input data.
    The second pointer also points to the start of a {vector_size}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
The kernel processes each element of the input vector, rounding it up to the nearest integer, and stores the result in the output vector.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -10.0,  # Minimum value for the input buffer
        "max": 10.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),  # Shape of the output buffer
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_vector):
    return np.ceil(input_vector).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")