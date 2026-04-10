import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
matrix_size = 16

function_signature = "void matmul_16x16_int32(int32_t* a, int32_t* b, int32_t* out)"

description = f"""
This HLS kernel performs matrix multiplication between two 16x16 matrices of 32-bit integers (int32_t).
The matrices are provided in row-major order. The function takes three parameters:
    1. A pointer to the first input matrix 'a' of size {matrix_size}x{matrix_size}.
    2. A pointer to the second input matrix 'b' of size {matrix_size}x{matrix_size}.
    3. A pointer to the output matrix 'out' where the result of the multiplication will be stored.
The result of the multiplication is clamped to fit within the range of an 32-bit integer.
"""

inputs_info = [
    {
        "name": "a",  # Name of the first input matrix
        "array_dimension": 2,  # Number of dimensions in the input matrix
        "shape": (matrix_size, matrix_size),  # Shape of the input matrix
        "total_elements": matrix_size * matrix_size,  # Total number of elements in the input matrix
        "dtype": "int32",
        "iotype": "in",  # Input matrix for the kernel
        "min": -1024,  # Minimum value for the input matrix
        "max": 1024,  # Maximum value for the input matrix
    },
    {
        "name": "b",  # Name of the second input matrix
        "array_dimension": 2,  # Number of dimensions in the input matrix
        "shape": (matrix_size, matrix_size),  # Shape of the input matrix
        "total_elements": matrix_size * matrix_size,  # Total number of elements in the input matrix
        "dtype": "int32",
        "iotype": "in",  # Input matrix for the kernel
        "min": -1024,  # Minimum value for the input matrix
        "max": 1024,  # Maximum value for the input matrix
    },
    {
        "name": "out",  # Name of the output matrix
        "array_dimension": 2,  # Number of dimensions in the output matrix
        "shape": (matrix_size, matrix_size),  # Shape of the output matrix
        "total_elements": matrix_size * matrix_size,  # Total number of elements in the output matrix
        "dtype": "int32",
        "iotype": "out",  # Output matrix for the kernel
    }
]

def behavioral(a, b):
    # Matrix multiplication for int32, result clamped to int32.
    a_mat = a.reshape((16,16))
    b_mat = b.reshape((16,16))
    res = np.matmul(a_mat.astype(np.int32), b_mat.astype(np.int32))
    res = np.clip(res, -2**31, 2**31-1).astype(np.int32)
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