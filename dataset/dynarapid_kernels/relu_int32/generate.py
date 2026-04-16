import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
num_elements = 1024

function_signature = "void relu_int32(int32_t *in_buffer, int32_t* out_buffer)"

short_description = "This HLS kernel performs a ReLU activation on an int32 input vector of size num_elements."

description = f"""
This HLS kernel performs a ReLU (Rectified Linear Unit) activation on an input vector of 32-bit integer numbers.
Parameters to the function include two int32_t pointers:
    The first pointer points to the start of a {num_elements}-element array of int32_t values to be used as input data.
    The second pointer also points to the start of a {num_elements}-element array of int32_t values to be modified and viewed as the output to the function.
The ReLU operation sets each element of the output buffer to the corresponding element of the input buffer if it is greater than zero, otherwise it sets it to zero.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_elements,),  # Shape of the input buffer
        "total_elements": num_elements,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -2**31,  # Minimum value for the input buffer
        "max": 2**31-1,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_elements,),
        "total_elements": num_elements,  # Total number of elements in the output buffer
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    return np.where(input_buffer > 0, input_buffer, 0).astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")