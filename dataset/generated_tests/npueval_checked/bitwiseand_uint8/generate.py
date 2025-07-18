import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 1024

function_signature = "void bitwiseand_uint8(ap_uint<8> *in_buffer1, ap_uint<8> *in_buffer2, ap_uint<8> *out_buffer)"

description = f"""
This HLS kernel performs a bitwise AND operation on two input vectors of 8-bit unsigned integers and stores the result in an output vector.
Parameters to the function include three ap_uint<8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_uint<8> values to be used as the first input data.
    The second pointer points to the start of a {num_values}-element array of ap_uint<8> values to be used as the second input data.
    The third pointer points to the start of a {num_values}-element array of ap_uint<8> values to be modified and viewed as the output of the function.
"""

inputs_info = [
    {
        "name": "in_buffer1",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "in_buffer2",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "uint8",
        "iotype": "out",
    }
]

def behavioral(in_buffer1, in_buffer2):
    return np.bitwise_and(in_buffer1, in_buffer2)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")