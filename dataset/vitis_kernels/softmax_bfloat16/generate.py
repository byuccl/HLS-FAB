import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

vector_size = 64  # Length of the input vector

function_signature = "void softmax_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel implements the softmax function on a vector of bfloat16 values.
    First parameter: Pointer to the input vector of {vector_size} ap_float<16,8> values.
    Second parameter: Pointer to the output vector of {vector_size} ap_float<16,8> values.
The kernel computes the softmax of the input vector, which involves exponentiating each element,
normalizing by the sum of all exponentials, and storing the result in the output vector.
The input and output vectors are expected to be of length {vector_size}.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "bfloat16",
        "iotype": "in",  # Input vector for the kernel
        "min": -3,  # Minimum value for the input vector
        "max": 3.0,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "output_vector",  # Name of the output vector
        "array_dimension": 1,  # Number of dimensions in the output vector
        "shape": (vector_size,),  # Shape of the output vector
        "total_elements": vector_size,  # Total number of elements in the output vector
        "dtype": "bfloat16",
        "iotype": "out",  # Output vector for the kernel
    }  # Output vector for the kernel
]

def behavioral(input_vector):
    x = input_vector - np.max(input_vector)
    exp_x = np.exp(x)
    return (exp_x / np.sum(exp_x)).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)


pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,

)

pc.write_json("kernel.json")