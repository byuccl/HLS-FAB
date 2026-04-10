import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 256

function_signature = "void hardsigmoid_int32(int32_t *input_vector, int32_t *output_vector)"

description = f"""
This HLS kernel applies the hardsigmoid activation function on a vector of 32-bit integer values.
The hardsigmoid is computed as: y = max(0, min(1, x/6 + 0.5)), where x is the input value.
The result is then rescaled and quantized into an 32-bit integer output with the range [0, 65535].
Parameters to the function include two int32_t pointers:
    The first pointer points to the start of a {VECTOR_SIZE}-element array of int32_t values to be used as input data.
    The second pointer points to the start of a {VECTOR_SIZE}-element array of int32_t values to be modified and viewed as the output of the function.
"""

inputs_info = [
    {
        "name": "input_vector",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int32",
        "iotype": "in",
        "min": -65536,  # Minimum value for the input buffer
        "max": 65535,  # Maximum value for the input buffer
    },
    {
        "name": "output_vector",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(input_vector):
    x = input_vector.astype(np.float32)
    y = x / 6.0 + 0.5
    y = np.clip(y, 0.0, 1.0)
    out = np.round(y * 65535.0).clip(0, 65535).astype(np.int32)
    return out

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")