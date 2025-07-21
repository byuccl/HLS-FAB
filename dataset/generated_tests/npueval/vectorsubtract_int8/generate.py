import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 512

function_signature = "void vectorsubtract_int8(ap_int<8> *in_buffer1, ap_int<8> *in_buffer2, ap_int<8> *out_buffer)"

description = f"""
This HLS kernel performs a vector subtraction operation on two input vectors of 8-bit integer numbers.
Parameters to the function include three ap_int<8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_int<8> values to be used as the first input vector.
    The second pointer points to the start of a {num_values}-element array of ap_int<8> values to be used as the second input vector.
    The third pointer points to the start of a {num_values}-element array of ap_int<8> values to store the result of the subtraction.
The kernel computes the element-wise subtraction of the second input vector from the first input vector and stores the result in the output vector.
"""

inputs_info = [
    {
        "name": "in_buffer1",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -128,  # Minimum value for the input buffer
        "max": 127,  # Maximum value for the input buffer
    },
    {
        "name": "in_buffer2",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -128,  # Minimum value for the input buffer
        "max": 127,  # Maximum value for the input buffer
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "int8",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(in_buffer1, in_buffer2):
    return (in_buffer1 - in_buffer2).astype(np.int8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")