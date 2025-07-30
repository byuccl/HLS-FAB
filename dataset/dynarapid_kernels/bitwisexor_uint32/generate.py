import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 1024

function_signature = "void bitwisexor_uint32(uint32_t *in_buffer1, uint32_t *in_buffer2, uint32_t *out_buffer)"

description = f"""
This HLS kernel implements a bitwise XOR operation on two input vectors of 32-bit unsigned integers.
Parameters to the function include three uint32_t pointers:
    The first pointer points to the start of a {num_values}-element array of uint32_t values to be used as the first input data.
    The second pointer points to the start of a {num_values}-element array of uint32_t values to be used as the second input data.
    The third pointer points to the start of a {num_values}-element array of uint32_t values to be modified and viewed as the output of the function.
The kernel computes the bitwise XOR of corresponding elements from the two input buffers and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer1",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "uint32",
        "iotype": "in",
        "min": 0,
        "max": 2**32-1,
    },
    {
        "name": "in_buffer2",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "uint32",
        "iotype": "in",
        "min": 0,
        "max": 2**32-1,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "uint32",
        "iotype": "out",
    }
]

def behavioral(in_buffer1, in_buffer2):
    return np.bitwise_xor(in_buffer1, in_buffer2)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")