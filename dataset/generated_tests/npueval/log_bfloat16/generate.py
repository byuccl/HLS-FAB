import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void log_bfloat16(ap_float<16,8> *input_buffer, ap_float<16,8> *output_buffer)"

description = f"""
This HLS kernel implements the natural logarithm function on a vector of bfloat16 values.
    First parameter: Pointer to the input buffer of {num_values} ap_float<16,8> values.
    Second parameter: Pointer to the output buffer of ap_float<16,8> values.
The kernel computes the natural logarithm of each element in the input buffer and stores the result in the output buffer.
The input buffer is expected to contain positive values to avoid domain errors in the logarithm calculation.
"""

inputs_info = [
    {
        "name": "input_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0.001,  # Minimum value for the input buffer
        "max": 100.0,  # Maximum value for the input buffer
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
    return np.log(input_buffer).astype(bfloat16)

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