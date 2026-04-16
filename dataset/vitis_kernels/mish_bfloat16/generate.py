import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void mish_bfloat16(ap_float<16,8> *input_buffer, ap_float<16,8> *output_buffer)"

short_description = "This HLS kernel computes the elementwise Mish activation, Mish(x) = x * tanh(ln(1 + exp(x))), for a bfloat16 input vector of length 256, storing results in the bfloat16 output buffer."

description = f"""
This HLS kernel computes the elementwise Mish activation function for a bfloat16 input vector of length {num_values}.
The Mish activation is defined as Mish(x) = x * tanh(ln(1 + exp(x))).
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {num_values}-element array of ap_float<16,8> values to store the results of the Mish activation.
"""

inputs_info = [
    {
        "name": "input_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -4,  # Minimum value for the input buffer
        "max": 4.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    x = input_buffer
    softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)
    mish = x * np.tanh(softplus)
    return mish.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)


pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")