import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void sign_int32(int32_t *input_vector, int32_t *output_vector)"

description = f"""
This HLS kernel computes the elementwise sign of a vector of 32-bit integer values. 
Parameters to the function include two int32_t pointers:
    The first pointer points to the start of a {vector_size}-element array of int32_t values to be used as input data.
    The second pointer also points to the start of a {vector_size}-element array of int32_t values to be modified and viewed as the output to the function.
For each input value: if positive, the output is 1; if negative, the output is -1; otherwise, the output is 0.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -2**31,  # Minimum value for the input buffer
        "max": 2**31-1,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_vector):
    result = np.zeros_like(input_vector)
    result[input_vector > 0] = 1
    result[input_vector < 0] = -1
    return result.astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")