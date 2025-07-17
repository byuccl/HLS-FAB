import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void log10_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

description = f"""
This HLS kernel computes the base-10 logarithm (log10) elementwise for a bfloat16 input vector.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
For any negative or zero input values, the output will be set to -infinity.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0.001,  # Minimum value for the input buffer
        "max": 100.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    out = np.where(input_buffer > 0, np.log10(input_buffer.astype(np.float32)), -np.inf)
    return out.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

tolerances = {"atol": 0.02, "rtol": 0.02}

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
    tolerances=tolerances
)

pc.write_json("kernel.json")