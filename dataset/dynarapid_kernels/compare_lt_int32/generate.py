import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_elements = 256

function_signature = "void compare_lt_int32(int32_t* input_a, int32_t* input_b, int32_t* output)"

description = f"""
This HLS kernel performs an element-wise comparison between two input vectors of 32-bit integers, each of length {num_elements}. 
For each index i, the kernel outputs 1 if input_a[i] is less than input_b[i], otherwise it outputs 0. 
The output is stored in a third vector of 32-bit integers.
Parameters:
    - First parameter: Pointer to the first input buffer of {num_elements} int32_t values.
    - Second parameter: Pointer to the second input buffer of {num_elements} int32_t values.
    - Third parameter: Pointer to the output buffer of {num_elements} int32_t values, where the comparison results are stored.
"""

inputs_info = [
    {
        "name": "input_a",
        "array_dimension": 1,
        "shape": (num_elements,),
        "total_elements": num_elements,
        "dtype": "int32",
        "iotype": "in",
        "min": -100,
        "max": 100,
    },
    {
        "name": "input_b",
        "array_dimension": 1,
        "shape": (num_elements,),
        "total_elements": num_elements,
        "dtype": "int32",
        "iotype": "in",
        "min": -100,
        "max": 100,
    },
    {
        "name": "output",
        "array_dimension": 1,
        "shape": (num_elements,),
        "total_elements": num_elements,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(input_a, input_b):
    return (input_a < input_b).astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")