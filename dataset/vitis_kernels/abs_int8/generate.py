import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
num_values = 1024

function_signature = "void abs_int8(ap_int<8> *in_buffer, ap_int<8>* out_buffer)"

description = f"""
This HLS kernel implements a function to calculate the absolute value of a vector of 8-bit integer numbers.
Parameters to the function include two ap_int<8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_int<8> values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of ap_int<8> values to be modified and viewed as the output to the function.
The kernel processes each element of the input buffer, computes its absolute value, and stores the result in the corresponding position of the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int8",
        "iotype": "in",
        "min": -128,
        "max": 127,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int8",
        "iotype": "out",
    }
]

def behavioral(input_vector):
    return np.abs(input_vector).astype(np.int8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")