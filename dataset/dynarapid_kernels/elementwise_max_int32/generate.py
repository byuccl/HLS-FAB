import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 512

function_signature = "void elementwise_max_int32(ap_int<32> *in_buffer1, ap_int<32> *in_buffer2, ap_int<32> *out_buffer)"

description = f"""
This HLS kernel computes the elementwise maximum between two vectors of 32-bit integer numbers.
Parameters to the function include three ap_int<32> pointers:
    The first pointer points to the start of a {VECTOR_SIZE}-element array of ap_int<32> values to be used as the first input vector.
    The second pointer points to the start of a {VECTOR_SIZE}-element array of ap_int<32> values to be used as the second input vector.
    The third pointer points to the start of a {VECTOR_SIZE}-element array of ap_int<32> values to be modified and viewed as the output vector.
The kernel compares each element of the two input vectors and stores the maximum value in the corresponding position of the output vector.
"""

inputs_info = [
    {
        "name": "in_buffer1",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int32",
        "iotype": "in",
        "min": -2**31,
        "max": 2**31-1,
    },
    {
        "name": "in_buffer2",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int32",
        "iotype": "in",
        "min": -2**31,
        "max": 2**31-1,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(in_buffer1, in_buffer2):
    return np.maximum(in_buffer1, in_buffer2).astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")