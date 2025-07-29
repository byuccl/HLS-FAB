import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void bitcount_uint16(ap_uint<16> *input_vector, ap_uint<16> *output_vector)"

description = f"""
This HLS kernel counts the number of 1-bits (Hamming weight) in each entry of a 16-bit unsigned integer vector.
Parameters to the function include two ap_uint<16> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_uint<16> values to be used as input data.
    The second pointer also points to the start of a {vector_size}-element array of ap_uint<16> values to be modified and viewed as the output to the function.
The output vector will contain the bit count for each corresponding input value.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "uint16",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0,  # Minimum value for the input buffer
        "max": 65535,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "uint16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_vector):
    input_vector = np.asarray(input_vector, dtype=np.uint16)
    arr8 = input_vector.view(np.uint8).reshape(-1, 2)
    bits = np.unpackbits(arr8, axis=1)
    return np.sum(bits, axis=1).astype(np.uint16)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")