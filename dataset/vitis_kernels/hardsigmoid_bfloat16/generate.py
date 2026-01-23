import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void hardsigmoid_bfloat16(ap_float<16,8> *in_buffer, ap_float<16,8> *out_buffer)"

description = f"""
This HLS kernel performs a hard sigmoid activation on a bfloat16 input vector.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
The kernel applies the hard sigmoid function, which is defined as clip(0.2 * x + 0.5, 0, 1), to each element of the input buffer and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -6,  # Minimum value for the input buffer
        "max": 6,  # Maximum value for the input buffer
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
    x = input_buffer
    return np.clip(bfloat16(0.2) * x + bfloat16(0.5), bfloat16(0), bfloat16(1)).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")