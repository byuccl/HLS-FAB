import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void cumsum_bfloat16(ap_float<16,8> *in_buffer, ap_float<16,8> *out_buffer)"

description = f"""
This HLS kernel computes the elementwise cumulative sum (cumsum) of a bfloat16 input vector of length {num_values}.
Each output element is the sum of all input elements up to and including that position.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    return np.cumsum(input_buffer.astype(bfloat16)).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)


pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")