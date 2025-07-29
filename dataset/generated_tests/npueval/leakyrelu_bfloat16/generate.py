import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
num_elements = 256
alpha_value = 0.01

function_signature = "void leaky_relu_bfloat16(ap_float<16,8> *in_buffer, ap_float<16,8> *out_buffer)"

short_description = f"This HLS kernel performs a Leaky ReLU activation with an alpha value of {alpha_value} on a bfloat16 input vector of size {num_elements}."
description = f"""
This HLS kernel performs a Leaky ReLU activation on a bfloat16 input vector of size {num_elements}.
The Leaky ReLU function is defined as:
    f(x) = x if x > 0
    f(x) = alpha * x if x <= 0
where alpha is set to {alpha_value}.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_elements}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {num_elements}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_elements,),  # Shape of the input buffer
        "total_elements": num_elements,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3.0,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_elements,),  # Shape of the output buffer
        "total_elements": num_elements,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_buffer):
    return np.where(input_buffer > 0, input_buffer, input_buffer * bfloat16(alpha_value))

examples = generate_examples(inputs_info=inputs_info)


pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")