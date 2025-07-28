import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void sign_int8(ap_int<8> *input_vector, ap_int<8> *output_vector)"

description = f"""
This HLS kernel computes the elementwise sign of a vector of 8-bit integer values. 
Parameters to the function include two ap_int<8> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_int<8> values to be used as input data.
    The second pointer also points to the start of a {vector_size}-element array of ap_int<8> values to be modified and viewed as the output to the function.
For each input value: if positive, the output is 1; if negative, the output is -1; otherwise, the output is 0.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -128,  # Minimum value for the input buffer
        "max": 127,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "int8",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_vector):
    result = np.zeros_like(input_vector)
    result[input_vector > 0] = 1
    result[input_vector < 0] = -1
    return result.astype(np.int8)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")