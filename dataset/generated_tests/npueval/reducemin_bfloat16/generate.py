import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 64  # Number of elements in the input vector

function_signature = "void reducemin_bfloat16(ap_float<16,8> *in_buffer, ap_float<16,8>* out_buffer)"

description = f"""
This HLS kernel calculates the minimum value of an input vector of bfloat16 numbers and outputs the result to an output buffer.
Parameters to the function include:
    - A pointer to the input buffer containing {num_values} ap_float<16,8> values.
    - A pointer to the output buffer where the minimum value will be stored as a single ap_float<16,8> value.
The kernel iterates through the input buffer to find the minimum value and writes this value to the output buffer.
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
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 0,  # Number of dimensions in the output buffer
        "shape": (),  # Shape of the output buffer
        "total_elements": 1,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(in_buffer):
    return np.min(in_buffer).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")