import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void elementwise_min_int32(int32_t *input1, int32_t *input2, int32_t *output)"

description = f"""
This HLS kernel computes the elementwise minimum of two input vectors of 32-bit integer numbers.
Parameters to the function include three int32_t pointers:
    The first pointer points to the start of a {num_values}-element array of int32_t values to be used as the first input data.
    The second pointer points to the start of a {num_values}-element array of int32_t values to be used as the second input data.
    The third pointer points to the start of a {num_values}-element array of int32_t values to be modified and viewed as the output of the function.
The function computes the elementwise minimum of the two input arrays and stores the result in the output array.
"""

inputs_info = [
    {
        "name": "input1",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -2**31,
        "max": 2**31-1,
    },
    {
        "name": "input2",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -2**31,
        "max": 2**31-1,
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input1, input2):
    return np.minimum(input1, input2).astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")