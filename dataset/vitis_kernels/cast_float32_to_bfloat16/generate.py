import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void cast_float32_to_bfloat16(float *in_buffer, ap_float<16,8> *out_buffer)"

description = f"""
This HLS kernel casts a float32 input vector of length {num_values} to bfloat16, writing results to the output buffer.
    First parameter: Pointer to the input buffer of {num_values} float32 values.
    Second parameter: Pointer to the output buffer of {num_values} ap_float<16,8> values.
The kernel iterates over each element in the input buffer, converts it to bfloat16, and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "float32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -5.0,  # Minimum value for the input buffer
        "max": 5.0,  # Maximum value for the input buffer
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
    return input_buffer.astype(bfloat16)

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")