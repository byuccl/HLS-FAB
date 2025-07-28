import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void silu_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

description = f"""
This HLS kernel computes the SiLU (Sigmoid Linear Unit, also known as Swish) activation function, elementwise on a bfloat16 input vector of length {num_values}.
The SiLU function is defined as x * sigmoid(x), where sigmoid(x) = 1 / (1 + exp(-x)).
The input buffer is expected to be at least {num_values} ap_float<16,8> values, and the output buffer will contain the computed SiLU values.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    x = input_buffer.astype(np.float32)
    return (x * (1 / (1 + np.exp(-x)))).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")