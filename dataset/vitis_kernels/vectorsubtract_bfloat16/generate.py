import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vectors

function_signature = "void vectorsubtract_bfloat16(ap_float<16,8> *input_a, ap_float<16,8> *input_b, ap_float<16,8> *output)"

description = f"""
This HLS kernel performs elementwise subtraction between two bfloat16 input vectors, each of length {num_values}.
Parameters to the function include three ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as the first input vector.
    The second pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as the second input vector.
    The third pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output buffer.
The kernel computes the elementwise difference between the two input vectors and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "input_a",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -10,  # Minimum value for the input buffer
        "max": 10,  # Maximum value for the input buffer
    },  # First input buffer for the kernel
    {
        "name": "input_b",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -10,  # Minimum value for the input buffer
        "max": 10,  # Maximum value for the input buffer
    },  # Second input buffer for the kernel
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_a, input_b):
    return (input_a - input_b).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")