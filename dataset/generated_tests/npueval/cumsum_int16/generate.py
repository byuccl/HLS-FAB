import numpy as np
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void cumsum_int16(ap_int<16> *in_buffer, ap_int<16> *out_buffer)"

description = f"""
This HLS kernel computes the cumulative sum (cumsum) of a {num_values}-element vector of 16-bit integer numbers.
Parameters to the function include two ap_int<16> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_int<16> values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of ap_int<16> values to be modified and viewed as the output to the function.
The cumulative sum is calculated with saturation to the int16_t range on overflow.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -500,  # Minimum value for the input buffer
        "max": 500,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "int16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    acc = np.cumsum(input_buffer.astype(np.int32))
    acc = np.clip(acc, -32768, 32767)
    return acc.astype(np.int16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")