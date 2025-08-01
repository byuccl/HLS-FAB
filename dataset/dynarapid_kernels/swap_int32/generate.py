import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 1024

function_signature = "void swap_int32(int32_t *x_buffer, int32_t *y_buffer)"

short_description = "This HLS kernel swaps the values between two int32 buffers."

description = f"""
This HLS kernel swaps the values between two input vectors of 32-bit integer numbers.
Parameters to the function include two int32_t pointers:
    The first pointer points to the start of a {vector_size}-element array of int32_t values to be used as the first input vector.
    The second pointer points to the start of a {vector_size}-element array of int32_t values to be used as the second input vector.
The kernel swaps the corresponding elements of the two input buffers.
"""

inputs_info = [
    {
        "name": "x_buffer",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "int32",
        "iotype": "inout",
        "min": -2**31,
        "max": 2**31 - 1,
    },
    {
        "name": "y_buffer",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "int32",
        "iotype": "inout",
        "min": -2**31,
        "max": 2**31 - 1,
    }
]

def behavioral(x_buffer, y_buffer):
    x_buffer, y_buffer = y_buffer.copy(), x_buffer.copy()
    return x_buffer, y_buffer

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")