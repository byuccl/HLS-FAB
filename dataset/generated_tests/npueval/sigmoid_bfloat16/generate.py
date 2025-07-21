import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

vector_size = 256
exp_high = 88.3762626647949
exp_low = -87.3362626647949

function_signature = "void sigmoid_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel implements the sigmoid activation function on a vector of bfloat16 values.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
The kernel computes the sigmoid function for each element in the input vector, with special handling for values outside the range [{exp_low}, {exp_high}] to prevent overflow.
"""

inputs_info = [
    {
        "name": "input_vector",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "output_vector",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "bfloat16",
        "iotype": "out",
    }
]

def behavioral(input_vector):
    return (1 / (1 + np.exp(-input_vector).astype(bfloat16))).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
    tolerances=tolerances
)

pc.write_json("kernel.json")