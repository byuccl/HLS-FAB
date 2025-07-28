import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 256

function_signature = "void cast_int8_to_int32(ap_int<8>* input_vector, ap_int<32>* output_vector)"

description = f"""
This HLS kernel casts each element of a signed 8-bit integer input vector to a signed 32-bit integer, storing the result in the output buffer.
Parameters to the function include two pointers:
    The first pointer points to the start of a {VECTOR_SIZE}-element array of ap_int<8> values to be used as input data.
    The second pointer points to the start of a {VECTOR_SIZE}-element array of ap_int<32> values to be modified and viewed as the output of the function.
The casting operation preserves the sign of the input values.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (VECTOR_SIZE,),  # Shape of the input buffer
        "total_elements": VECTOR_SIZE,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -128,  # Minimum value for the input buffer
        "max": 127,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,  # Total number of elements in the output buffer
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_vector):
    return input_vector.astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")