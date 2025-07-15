import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void elementwise_min_int8(ap_int<8> *input1, ap_int<8> *input2, ap_int<8> *output)"

description = f"""
This HLS kernel computes the elementwise minimum of two input vectors of 8-bit integer numbers.
Parameters to the function include three ap_int<8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_int<8> values to be used as the first input data.
    The second pointer points to the start of a {num_values}-element array of ap_int<8> values to be used as the second input data.
    The third pointer points to the start of a {num_values}-element array of ap_int<8> values to be modified and viewed as the output of the function.
The kernel computes the elementwise minimum of the two input arrays and stores the result in the output array.
"""

inputs_info = [
    {
        "name": "input1",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int8",
        "iotype": "in",
        "min": -128,
        "max": 127,
    },
    {
        "name": "input2",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int8",
        "iotype": "in",
        "min": -128,
        "max": 127,
    },
    {
        "name": "output",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int8",
        "iotype": "out",
    }
]

def behavioral(input1, input2):
    return np.minimum(input1, input2).astype(np.int8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")