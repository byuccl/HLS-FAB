import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
vector_size = 256
pad_size = 2
pad_value = 0
padded_length = vector_size + 2 * pad_size

function_signature = "void pad1d_int32(int32_t *input_vector, int32_t *padded_vector)"

short_description = "A kernel that performs a 1D padding operation on an input vector with given padding size and padding value as runtime parameters."

description = f"""
This HLS kernel performs a 1D padding operation on an input vector of 32-bit integers. 
Parameters to the function include:
    - A pointer to the input vector of {vector_size} int32_t values.
    - A pointer to the output padded vector of int32_t values.
The output padded vector will have a length of {padded_length}, with the input vector centered and surrounded by pad_value.
"""

inputs_info = [
    {
        "name": "input_vector",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "int32",
        "iotype": "in",
        "min": -10,
        "max": 10,
    },
    {
        "name": "padded_vector",
        "array_dimension": 1,
        "shape": (padded_length,),
        "total_elements": padded_length,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(input_vector):
    padded_length = len(input_vector) + 2 * pad_size
    padded_vector = np.full(padded_length, pad_value, dtype=input_vector.dtype)
    padded_vector[pad_size:pad_size + len(input_vector)] = input_vector
    return padded_vector

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")