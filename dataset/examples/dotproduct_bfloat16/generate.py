import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_types, generate_examples

vector_size = 64  # Size of the input vectors

function_signature = "void dotproduct_bfloat16(ap_float<16,8> *vector1, ap_float<16,8> *vector2, ap_float<16,8> *result)"

short_description = "An HLS kernel that performs a dot product operation on two bfloat16 input vectors of the same size."

description = f"""
An HLS kernel that performs a dot product operation on two input vectors of bfloat16 values.
    First parameter: Pointer to the first input vector of {vector_size} ap_float<16,8> values.
    Second parameter: Pointer to the second input vector of {vector_size} ap_float<16,8> values.
    Third parameter: Pointer to a single ap_float<16,8> value to store the result of the dot product.
The kernel computes the dot product of the two input vectors and stores the result in the provided result pointer.
"""

inputs_info = [
    {
        "name": "vector1",  # Name of the first input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "bfloat16",
        "iotype": "in",  # Input vector for the kernel
        "min": -3,  # Minimum value for the input vector
        "max": 3.0,  # Maximum value for the input vector
    },
    {
        "name": "vector2",  # Name of the second input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "bfloat16",
        "iotype": "in",  # Input vector for the kernel
        "min": -3,  # Minimum value for the input vector
        "max": 3.0,  # Maximum value for the input vector
    },
    {
        "name": "result",  # Name of the result
        "array_dimension": 0,  # Number of dimensions in the result
        "shape": (),  # Shape of the result
        "total_elements": 1,  # Total number of elements in the result
        "dtype": "bfloat16",
        "iotype": "out",  # Output for the kernel
    }
]

def behavioral(vector1, vector2):
    return np.array([np.dot(vector1, vector2)], dtype=bfloat16)

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")