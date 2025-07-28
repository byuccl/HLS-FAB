import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 1024
offset = 2

function_signature = "void add_offset_uint8(ap_uint<8> *in_buffer, ap_uint<8>* out_buffer, ap_uint<8> offset)"

description = f"""
This HLS kernel adds a constant offset to each element of an input vector containing 8-bit unsigned integer data.
Parameters to the function include:
    - A pointer to the input buffer of {num_values} ap_uint<8> values.
    - A pointer to the output buffer of {num_values} ap_uint<8> values.
    - An ap_uint<8> offset value to be added to each element of the input buffer.
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
        "max": 250,  # Maximum value for the input buffer
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "uint8",
        "iotype": "out",  # Output buffer for the kernel
    },
    {
        "name": "offset",  # Name of the offset parameter
        "array_dimension": 0,  # No dimensions for a single value
        "shape": (),  # Shape of the offset parameter
        "total_elements": 1,  # Total number of elements in the offset parameter
        "dtype": "uint8",  # Data type of the offset
        "iotype": "in",  # Input parameter for the kernel
        "min": 2,  # Minimum value for the offset
        "max": 5,  # Maximum value for the offset
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