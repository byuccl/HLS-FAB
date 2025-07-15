import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void compare_equal_bfloat16(ap_float<16,8> *input_a, ap_float<16,8> *input_b, ap_uint<8> *output_mask)"

description = f"""
This HLS kernel performs an element-wise comparison of two input vectors of bfloat16 values for equality.
Parameters to the function include:
    - First parameter: Pointer to the first input buffer of {vector_size} ap_float<16,8> values.
    - Second parameter: Pointer to the second input buffer of {vector_size} ap_float<16,8> values.
    - Third parameter: Pointer to the output buffer of ap_uint<8> values, which will store the result of the comparison.
The kernel writes a uint8_t mask output where each element is set to 1 if the corresponding elements in the input vectors are equal, and 0 otherwise.
"""
#Shawn uses the same exact input arrays except one different value at a random index.
inputs_info = [
    {
        "name": "input_a",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "input_b",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "output_mask",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "uint8",
        "iotype": "out",
    }
]

def behavioral(input_a, input_b):
    return (input_a == input_b).astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")