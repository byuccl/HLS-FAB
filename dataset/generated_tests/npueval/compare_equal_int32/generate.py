import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 256

function_signature = "void compare_equal_int32(ap_int<32>* a, ap_int<32>* b, ap_int<8>* out)"

description = f"""
This HLS kernel performs an element-wise comparison of two input vectors of 32-bit integers for equality.
Parameters to the function include:
    - First parameter: Pointer to the first input buffer of {VECTOR_SIZE} ap_int<32> values.
    - Second parameter: Pointer to the second input buffer of {VECTOR_SIZE} ap_int<32> values.
    - Third parameter: Pointer to the output buffer of {VECTOR_SIZE} ap_int<8> values.
The kernel compares each element of the two input buffers and writes 1 to the output buffer if the elements are equal, otherwise writes 0.
"""

inputs_info = [
    {
        "name": "a",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (VECTOR_SIZE,),  # Shape of the input buffer
        "total_elements": VECTOR_SIZE,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1000,  # Minimum value for the input buffer
        "max": 1000,  # Maximum value for the input buffer
    },
    {
        "name": "b",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (VECTOR_SIZE,),  # Shape of the input buffer
        "total_elements": VECTOR_SIZE,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1000,  # Minimum value for the input buffer
        "max": 1000,  # Maximum value for the input buffer
    },
    {
        "name": "out",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (VECTOR_SIZE,),  # Shape of the output buffer
        "total_elements": VECTOR_SIZE,  # Total number of elements in the output buffer
        "dtype": "uint8",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(a, b):
    return (a == b).astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")