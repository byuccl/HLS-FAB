import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

vector_size = 256  # Length of the input vectors

function_signature = "void divide_bfloat16(ap_float<16,8> *a, ap_float<16,8> *b, ap_float<16,8> *out)"

description = f"""
This HLS kernel performs element-wise division of two input vectors of bfloat16 numbers.
Parameters to the function include three ap_float<16,8> pointers:
    The first pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be used as the dividend.
    The second pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to be used as the divisor.
    The third pointer points to the start of a {vector_size}-element array of ap_float<16,8> values to store the quotient.
The kernel divides each element of the first input vector by the corresponding element of the second input vector.
If an element in the divisor vector is zero, the corresponding element in the output vector is set to 0.0.
"""

inputs_info = [
    {
        "name": "a",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -5.0,  # Minimum value for the input buffer
        "max": 5.0,  # Maximum value for the input buffer
    },
    {
        "name": "b",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -5.0,  # Minimum value for the input buffer
        "max": 5.0,  # Maximum value for the input buffer
    },
    {
        "name": "out",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),  # Shape of the output buffer
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(a, b):
    out = np.divide(a, b, out=np.zeros_like(a), where=b!=0)
    return out.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")