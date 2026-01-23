import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void tan_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

description = f"""
This HLS kernel computes the elementwise tangent (tan) function for a bfloat16 input vector of size {num_values}.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
The input values are clipped to the range [-15, 15] to ensure numerical stability.
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
        "shape": (num_values,),
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_array):
    x = np.clip(input_array.astype(np.float32), -15, 15)
    return np.tan(x).astype(bfloat16)

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