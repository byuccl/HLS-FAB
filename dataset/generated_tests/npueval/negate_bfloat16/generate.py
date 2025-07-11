import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void negate_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel negates each element of a bfloat16 input vector and writes the result to an output vector.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be used as input data.
    The second pointer also points to the start of a {vector_size}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
The kernel iterates over each element of the input vector, negates it, and stores the result in the corresponding position of the output vector.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "bfloat16",
        "iotype": "in",  # Input vector for the kernel
        "min": -4.0,  # Minimum value for the input vector
        "max": 4.0,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "output_vector",  # Name of the output vector
        "array_dimension": 1,  # Number of dimensions in the output vector
        "shape": (vector_size,),
        "total_elements": vector_size,  # Total number of elements in the output vector
        "dtype": "bfloat16",
        "iotype": "out",  # Output vector for the kernel
    }  # Output vector for the kernel
]

def behavioral(input_vector):
    return (-input_vector).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")