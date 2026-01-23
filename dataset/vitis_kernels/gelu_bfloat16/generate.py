import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
vector_size = 256
sqrt_2_over_pi = 0.7978845608028654
coeff = 0.044715

function_signature = "void gelu_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel implements the GELU (Gaussian Error Linear Unit) activation function using an approximation formula.
The function takes two parameters:
    1. A pointer to the input vector of {vector_size} ap_float<16,8> values.
    2. A pointer to the output vector of ap_float<16,8> values.
The GELU function is applied element-wise to the input vector, and the results are stored in the output vector.
The approximation used is based on the hyperbolic tangent function for computational efficiency.
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
    x = input_vector
    cdf = 0.5 * (1.0 + np.tanh(sqrt_2_over_pi * (x + coeff * np.power(x, 3))))
    return (x * cdf).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")