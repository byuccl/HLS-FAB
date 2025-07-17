import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 64  # Length of the input vector

function_signature = "void exp_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel implements the exponential function on a vector of bfloat16 numbers.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
The kernel computes the exponential of each element in the input buffer and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_vector):
    return np.exp(input_vector).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
    tolerances=tolerances
)

pc.write_json("kernel.json")