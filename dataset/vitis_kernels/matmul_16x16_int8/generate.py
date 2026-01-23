import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
matrix_size = 16

function_signature = "void matmul_16x16_int8(ap_int<8>* a, ap_int<8>* b, ap_int<8>* out)"

description = f"""
This HLS kernel performs matrix multiplication between two 16x16 matrices of 8-bit integers (ap_int<8>).
The matrices are provided in row-major order. The function takes three parameters:
    1. A pointer to the first input matrix 'a' of size {matrix_size}x{matrix_size}.
    2. A pointer to the second input matrix 'b' of size {matrix_size}x{matrix_size}.
    3. A pointer to the output matrix 'out' where the result of the multiplication will be stored.
The result of the multiplication is clamped to fit within the range of an 8-bit integer.
"""

inputs_info = [
    {
        "name": "a",  # Name of the first input matrix
        "array_dimension": 2,  # Number of dimensions in the input matrix
        "shape": (matrix_size, matrix_size),  # Shape of the input matrix
        "total_elements": matrix_size * matrix_size,  # Total number of elements in the input matrix
        "dtype": "int8",
        "iotype": "in",  # Input matrix for the kernel
        "min": -10,  # Minimum value for the input matrix
        "max": 10,  # Maximum value for the input matrix
    },
    {
        "name": "b",  # Name of the second input matrix
        "array_dimension": 2,  # Number of dimensions in the input matrix
        "shape": (matrix_size, matrix_size),  # Shape of the input matrix
        "total_elements": matrix_size * matrix_size,  # Total number of elements in the input matrix
        "dtype": "int8",
        "iotype": "in",  # Input matrix for the kernel
        "min": -10,  # Minimum value for the input matrix
        "max": 10,  # Maximum value for the input matrix
    },
    {
        "name": "out",  # Name of the output matrix
        "array_dimension": 2,  # Number of dimensions in the output matrix
        "shape": (matrix_size, matrix_size),  # Shape of the output matrix
        "total_elements": matrix_size * matrix_size,  # Total number of elements in the output matrix
        "dtype": "int8",
        "iotype": "out",  # Output matrix for the kernel
    }
]

def behavioral(a, b):
    # Matrix multiplication for int8, result clamped to int8.
    a = a.reshape((matrix_size, matrix_size))
    b = b.reshape((matrix_size, matrix_size))
    res = np.zeros((matrix_size,matrix_size), np.int32)
    res = np.matmul(a.astype(np.int32), b.astype(np.int32))
    res = np.clip(res, -128, 127).astype(np.int8)
    return res.flatten()

examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")