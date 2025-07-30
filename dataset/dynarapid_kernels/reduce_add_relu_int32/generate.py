import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_elements = 128

function_signature = "void reduce_add_relu_int32(int32_t *in_buffer, int32_t* out_buffer)"

short_description = "This HLS kernel sums all 128 elements of an int32_t input buffer, applies a ReLU (sets negative sums to zero), and outputs the saturated result as a single int32_t to out_buffer."

description = f"""
This HLS kernel performs a reduction operation by summing all {num_elements} elements of an 32-bit integer input buffer.
The sum is then passed through a ReLU activation function, which sets any negative sums to zero.
Finally, the result is saturated to fit within the range of an 32-bit integer and stored in the output buffer.
Parameters to the function include:
    - A pointer to the input buffer of {num_elements} int32_t values.
    - A pointer to a single int32_t value to store the output.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_elements,),  # Shape of the input buffer
        "total_elements": num_elements,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -100,  # Minimum value for the input buffer
        "max": 100,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 0,  # Number of dimensions in the output buffer
        "shape": (),  # Shape of the output buffer
        "total_elements": 1,  # Total number of elements in the output buffer
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    s = np.sum(input_buffer.astype(np.int32))
    s = max(s, 0)
    s = np.clip(s, -2**31, 2**31-1)
    return np.int32(s)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")