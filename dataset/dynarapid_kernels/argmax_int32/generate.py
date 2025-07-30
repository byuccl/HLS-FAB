import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void argmax_int32(int32_t *input_vector, uint32_t *result)"

description = f"""
This HLS kernel calculates the argmax, which is the index of the largest element, of a given int32 vector of length {vector_size}.
Parameters to the function include:
    - A pointer to the input vector of {vector_size} int32_t values.
    - A pointer to a single uint32_t element that will store the result, which is the index of the maximum value in the input vector.
The kernel iterates through the input vector to find the maximum value and its corresponding index, storing the index in the result.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": -32768,  # Minimum value for the input vector
        "max": 32767,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "result",  # Name of the result
        "array_dimension": 0,  # Number of dimensions in the result
        "shape": (),  # Shape of the result
        "total_elements": 1,  # Total number of elements in the result
        "dtype": "uint32",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

def behavioral(input_vector):
    # Return as shape (1,) array for correct output serialization
    return np.argmax(input_vector).astype(np.uint32)

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")