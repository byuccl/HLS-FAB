import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

vector_size = 64  # Length of the input vector

function_signature = "void argmax_bfloat16(ap_float<16,8> *input_vector, ap_uint<32> *result)"

description = f"""
This HLS kernel calculates the index of the maximum value (argmax) in a vector of bfloat16 numbers.
Parameters to the function include:
    - A pointer to the input vector of {vector_size} ap_float<16,8> values.
    - A pointer to a single ap_uint<32> value where the index of the maximum value will be stored.
The kernel iterates through the input vector to find the maximum value and its corresponding index, which is then stored in the result.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "bfloat16",
        "iotype": "in",  # Input vector for the kernel
        "min": -3.0,  # Minimum value for the input vector
        "max": 3.0,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "result",  # Name of the result
        "array_dimension": 0,  # Number of dimensions in the result
        "shape": (),  # Shape of the result
        "total_elements": 1,  # Total number of elements in the result
        "dtype": "uint32",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

def behavioral(input_vector):
    return np.argmax(input_vector).astype(np.uint32)

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")