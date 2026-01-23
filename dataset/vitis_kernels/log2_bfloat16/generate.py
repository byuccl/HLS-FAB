import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void log2_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

description = f"""
This HLS kernel computes the base-2 logarithm (log2) of each element in an input vector of bfloat16 values.
The function takes two parameters:
    1. A pointer to the input buffer containing {num_values} ap_float<16,8> values.
    2. A pointer to the output buffer where the results will be stored.
For zero or negative inputs, the function returns -inf in the output buffer.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0.001,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_buffer):
    with np.errstate(divide='ignore'):
        out = np.log2(input_buffer.astype(np.float32))
        out = np.where(input_buffer <= 0, -np.inf, out)
    return out.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)


pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")