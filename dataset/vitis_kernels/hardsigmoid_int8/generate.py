import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 256

function_signature = "void hardsigmoid_int8(ap_int<8> *input_vector, ap_int<8> *output_vector)"

description = f"""
This HLS kernel applies the hardsigmoid activation function on a vector of 8-bit integer values.
The hardsigmoid is computed as: y = max(0, min(1, x/6 + 0.5)), where x is the input value.
The result is then rescaled and quantized into an 8-bit integer output with the range [0, 127].
Parameters to the function include two ap_int<8> pointers:
    The first pointer points to the start of a {VECTOR_SIZE}-element array of ap_int<8> values to be used as input data.
    The second pointer points to the start of a {VECTOR_SIZE}-element array of ap_int<8> values to be modified and viewed as the output of the function.
"""

inputs_info = [
    {
        "name": "input_vector",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int8",
        "iotype": "in",
        "min": -128,
        "max": 128,
    },
    {
        "name": "output_vector",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int8",
        "iotype": "out",
    }
]

def behavioral(input_vector):
    x = input_vector.astype(np.float32)
    y = x / 6.0 + 0.5
    y = np.clip(y, 0.0, 1.0)
    out = np.round(y * 127.0).clip(0, 127).astype(np.int8)
    return out

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")