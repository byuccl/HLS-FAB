import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
vec_size = 256

function_signature = "void elementwise_min_float32(float *in_buffer_a, float *in_buffer_b, float *out_buffer)"

short_description = "This HLS kernel performs an elementwise minimum operation between two input float32 vectors, writing the minimum of each pair to the output buffer."

description = f"""
This HLS kernel performs an elementwise minimum operation between two input vectors of float32 values.
Parameters to the function include three float pointers:
    The first pointer points to the start of a {vec_size}-element array of float values to be used as the first input data.
    The second pointer points to the start of a {vec_size}-element array of float values to be used as the second input data.
    The third pointer points to the start of a {vec_size}-element array of float values to be modified and viewed as the output of the function.
The kernel computes the elementwise minimum of each pair of elements from the two input buffers and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer_a",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vec_size,),  # Shape of the input buffer
        "total_elements": vec_size,  # Total number of elements in the input buffer
        "dtype": "float32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1,  # Minimum value for the input buffer
        "max": 1,  # Maximum value for the input buffer
    },
    {
        "name": "in_buffer_b",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vec_size,),  # Shape of the input buffer
        "total_elements": vec_size,  # Total number of elements in the input buffer
        "dtype": "float32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1,  # Minimum value for the input buffer
        "max": 1,  # Maximum value for the input buffer
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vec_size,),  # Shape of the output buffer
        "total_elements": vec_size,  # Total number of elements in the output buffer
        "dtype": "float32",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(in_buffer_a, in_buffer_b):
    return np.minimum(in_buffer_a, in_buffer_b).astype(float32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")