import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 1024
offset = 2

function_signature = "void add_offset_uint8(ap_int<8> *in_buffer, ap_int<8>* out_buffer, ap_int<8> offset)"

description = f"""
This HLS kernel adds a constant offset to each element of an input vector containing 8-bit unsigned integer data.
Parameters to the function include:
    - A pointer to the input buffer of {num_values} ap_int<8> values.
    - A pointer to the output buffer of {num_values} ap_int<8> values.
    - An ap_int<8> offset value to be added to each element of the input buffer.
The kernel processes each element of the input buffer by adding the offset and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "uint8",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0,  # Minimum value for the input buffer
        "max": 255,  # Maximum value for the input buffer
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "uint8",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_buffer, offset=offset):
    return (input_buffer + offset).astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")