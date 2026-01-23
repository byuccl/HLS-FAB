import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
vector_length = 1024

function_signature = "void scale_vector_uint16(uint16_t *input_vector, uint16_t scale_factor, uint16_t *output_vector)"

short_description = "This HLS kernel scales a vector of 1024 uint16 values by a scalar scale factor. It wraps around if is there is overflow."

description = f"""
This HLS kernel scales a vector of 1024 unsigned 16-bit integer values by a scalar scale factor.
Parameters to the function include:
    - A pointer to the input vector of {vector_length} uint16_t values.
    - A scalar uint16_t value as the scale factor.
    - A pointer to the output vector which will store the scaled result.
The kernel multiplies each element of the input vector by the scale factor and writes the result into the output vector. Naturally it does not protect against overflow.
"""

inputs_info = [
    {
        "name": "input_vector",
        "array_dimension": 1,
        "shape": (vector_length,),
        "total_elements": vector_length,
        "dtype": "uint16",
        "iotype": "in",
        "min": 0,
        "max": 4096,
    },
    {
        "name": "scale_factor",
        "array_dimension": 0,
        "shape": (),
        "total_elements": 1,
        "dtype": "uint16",
        "iotype": "in",
        "min": 2,
        "max": 16,
    },
    {
        "name": "output_vector",
        "array_dimension": 1,
        "shape": (vector_length,),
        "total_elements": vector_length,
        "dtype": "uint16",
        "iotype": "out",
    }
]

def behavioral(input_vector, scale_factor):
    return (input_vector * scale_factor).astype(np.uint16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")