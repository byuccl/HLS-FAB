import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

vector_size = 64  # Length of the input vectors

function_signature = "void complexabs_bfloat16(ap_float<16,8> *real_vector, ap_float<16,8> *imag_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel computes the absolute value of complex numbers. 
The kernel takes two input buffers and one output buffer:
    - The first input buffer, real_vector, contains the real parts of the complex numbers, with {vector_size} ap_float<16,8> values.
    - The second input buffer, imag_vector, contains the imaginary parts of the complex numbers, also with {vector_size} ap_float<16,8> values.
    - The output buffer, output_vector, will store the computed absolute values of the complex numbers, with {vector_size} ap_float<16,8> values.
The kernel calculates the absolute value using the formula sqrt(real^2 + imag^2) for each complex number.
"""

inputs_info = [
    {
        "name": "real_vector",  # Name of the real part input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "imag_vector",  # Name of the imaginary part input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),  # Shape of the output buffer
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(real_vector, imag_vector):
    return np.sqrt(real_vector**2 + imag_vector**2).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")