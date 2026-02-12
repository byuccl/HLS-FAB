import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 512  # Length of the input vector

function_signature = "void cast_bfloat16_to_float32(ap_float<16,8> *in_buffer, float *out_buffer)"

description = f"""
This HLS kernel performs an elementwise conversion of a bfloat16 input vector to a float32 output vector.
    First parameter: Pointer to the input buffer of {num_values} ap_float<16,8> values.
    Second parameter: Pointer to the output buffer of float values.
The kernel reads each element from the input buffer, casts it to float32, and writes it to the output buffer.
The input buffer is expected to be at least {num_values} ap_float<16,8> values.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3.0,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "float32",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    return input_buffer.astype(np.float32)

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")