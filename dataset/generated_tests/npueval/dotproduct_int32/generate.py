import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void dotproduct_int32(ap_int<32> *vector1, ap_int<32> *vector2, ap_int<32> *result)"

description = f"""
This HLS kernel performs a dot product operation on two input vectors of 32-bit integers.
Parameters to the function include:
    - The first parameter is a pointer to the first input vector of {vector_size} ap_int<32> values.
    - The second parameter is a pointer to the second input vector of {vector_size} ap_int<32> values.
    - The third parameter is a pointer to a single ap_int<32> value where the result of the dot product will be stored.
The kernel computes the sum of the products of corresponding elements from the two input vectors and stores the result in the provided result pointer.
"""

inputs_info = [
    {
        "name": "vector1",  # Name of the first input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": -10,  # Minimum value for the input vector
        "max": 10,  # Maximum value for the input vector
    },
    {
        "name": "vector2",  # Name of the second input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": -10,  # Minimum value for the input vector
        "max": 10,  # Maximum value for the input vector
    },
    {
        "name": "result",  # Name of the result
        "array_dimension": 0,  # Number of dimensions in the result
        "shape": (),  # Shape of the result
        "total_elements": 1,  # Total number of elements in the result
        "dtype": "int32",
        "iotype": "out",  # Output for the kernel
    }
]

def behavioral(vector1, vector2):
    return np.dot(vector1, vector2).astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")