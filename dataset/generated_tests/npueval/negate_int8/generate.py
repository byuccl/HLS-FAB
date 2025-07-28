import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void negate_int8(ap_int<8> *input_vector, ap_int<8> *output_vector)"

description = f"""
This HLS kernel negates each element in a vector of 8-bit integer numbers. 
Parameters to the function include two ap_int<8> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_int<8> values to be used as input data. 
    The second pointer also points to the start of a {vector_size}-element array of ap_int<8> values to be modified and viewed as the output to the function.
The kernel iterates over each element in the input vector, negates it, and stores the result in the corresponding position in the output vector.
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
        "max": 128,  # Maximum value for the input buffer
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

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    return (-input_vector).astype(np.int8)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")