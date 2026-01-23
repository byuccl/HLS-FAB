import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
num_elements = 1024

function_signature = "void relu_int8(ap_int<8> *in_buffer, ap_int<8>* out_buffer)"

short_description = "This HLS kernel performs a ReLU activation on an int8 input vector of size num_elements."

description = f"""
This HLS kernel performs a ReLU (Rectified Linear Unit) activation on an input vector of 8-bit integer numbers.
Parameters to the function include two ap_int<8> pointers:
    The first pointer points to the start of a {num_elements}-element array of ap_int<8> values to be used as input data.
    The second pointer also points to the start of a {num_elements}-element array of ap_int<8> values to be modified and viewed as the output to the function.
The ReLU operation sets each element of the output buffer to the corresponding element of the input buffer if it is greater than zero, otherwise it sets it to zero.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_elements,),  # Shape of the input buffer
        "total_elements": num_elements,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -128,  # Minimum value for the input buffer
        "max": 127,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_elements,),
        "total_elements": num_elements,  # Total number of elements in the output buffer
        "dtype": "int8",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    return np.where(input_buffer > 0, input_buffer, 0).astype(np.int8)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")