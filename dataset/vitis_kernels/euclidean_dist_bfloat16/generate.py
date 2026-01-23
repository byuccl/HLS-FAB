import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
num_values = 256  # Length of the input vectors

function_signature = "void euclidean_dist_bfloat16(ap_float<16,8>* input_a, ap_float<16,8>* input_b, ap_float<16,8>* output)"

description = f"""
This HLS kernel computes the Euclidean (L2) distance between two input vectors of bfloat16 values, each of length {num_values}.
Parameters to the function include:
    - The first parameter is a pointer to the first input buffer containing {num_values} ap_float<16,8> values.
    - The second parameter is a pointer to the second input buffer containing {num_values} ap_float<16,8> values.
    - The third parameter is a pointer to a single ap_float<16,8> value where the result of the Euclidean distance computation will be stored.
The kernel calculates the sum of squared differences between corresponding elements of the two input vectors, computes the square root of this sum, and writes the result to the output buffer.
"""

inputs_info = [
    {
        "name": "input_a",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3.0,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "input_b",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3.0,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 0,  # Number of dimensions in the output buffer
        "shape": (),  # Shape of the output buffer
        "total_elements": 1,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_a, input_b):
    acc = np.sum((input_a.astype(np.float32) - input_b.astype(np.float32))**2)
    return np.sqrt(acc).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")