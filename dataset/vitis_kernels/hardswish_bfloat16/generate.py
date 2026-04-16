import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void hardswish_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel implements the hard swish activation function on a vector of bfloat16 values.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be modified and viewed as the output of the function.
The hard swish function is defined as:
    hardswish(x) = x * ReLU6(x + 3) / 6
where ReLU6(x) = min(max(x, 0), 6).
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -6,  # Minimum value for the input buffer
        "max": 6.0,  # Maximum value for the input buffer
    },
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_vector):
    x = input_vector
    relu6 = np.minimum(np.maximum(x + 3, 0).astype(bfloat16), 6).astype(bfloat16)
    return (x * relu6 / 6).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
    tolerances=tolerances
)

pc.write_json("kernel.json")