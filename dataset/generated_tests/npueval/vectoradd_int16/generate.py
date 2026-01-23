import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void vectoradd_int16(ap_int<16> *in_buffer1, ap_int<16> *in_buffer2, ap_int<16> *out_buffer)"
short_description = "This is the implementation of an HLS kernel that performs a vector addition of 2 input vectors and stores the result in 1 output vector."

description = f"""
This HLS kernel performs a vector addition of two input vectors of 16-bit integer numbers and stores the result in an output vector.
Parameters to the function include three ap_int<16> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_int<16> values to be used as the first input vector.
    The second pointer points to the start of a {vector_size}-element array of ap_int<16> values to be used as the second input vector.
    The third pointer points to the start of a {vector_size}-element array of ap_int<16> values to be modified and viewed as the output vector.
The kernel computes the element-wise sum of the two input vectors and stores the result in the output vector.
"""

inputs_info = [
    {
        "name": "in_buffer1",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "int16",
        "iotype": "in",
        "min": -512,
        "max": 512,
    },
    {
        "name": "in_buffer2",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "int16",
        "iotype": "in",
        "min": -512,
        "max": 512,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "int16",
        "iotype": "out",
    }
]

def behavioral(in_buffer1, in_buffer2):
    return (in_buffer1 + in_buffer2).astype(np.int16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")