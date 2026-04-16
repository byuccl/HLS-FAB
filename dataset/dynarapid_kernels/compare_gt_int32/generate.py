import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void compare_gt_int32(int32_t *in_buffer0, int32_t *in_buffer1, int32_t *out_buffer)"

description = f"""
This HLS kernel compares two input vectors of 32-bit integers, each of length {num_values}, and produces an output vector of the same length.
For each element in the input vectors, the kernel checks if the element in the first input vector is greater than the corresponding element in the second input vector.
If the condition is true, the corresponding element in the output vector is set to 1; otherwise, it is set to 0.
The input and output vectors are represented using int32_t data types.
"""

inputs_info = [
    {
        "name": "in_buffer0",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -65536,  # Minimum value for the input buffer
        "max": 65535,  # Maximum value for the input buffer
    },
    {
        "name": "in_buffer1",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -65536,  # Minimum value for the input buffer
        "max": 65535,  # Maximum value for the input buffer
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(in_buffer0, in_buffer1):
    return (in_buffer0 > in_buffer1).astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")