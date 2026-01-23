import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
num_values = 1024

function_signature = "void abs_int32(int32_t *in_buffer, int32_t* out_buffer)"

description = f"""
This HLS kernel implements a function to calculate the absolute value of a vector of 32-bit integer numbers.
Parameters to the function include two int32_t pointers:
    The first pointer points to the start of a {num_values}-element array of int32_t values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of int32_t values to be modified and viewed as the output to the function.
The kernel processes each element of the input buffer, computes its absolute value, and stores the result in the corresponding position of the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int32",
        "iotype": "in",
        "min": -2**31,
        "max": 2**31 - 1,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(input_vector):
    return np.abs(input_vector).astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")