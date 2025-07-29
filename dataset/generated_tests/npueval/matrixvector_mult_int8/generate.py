import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants for matrix and vector dimensions
ROWS = 16
COLS = 16

function_signature = "void matrixvector_mult_int8(ap_int<8>* matrix, ap_int<8>* vector, ap_int<32>* result)"

short_description = f"A kernel that computes the matrix-vector multiplication of an int8 matrix ({ROWS}x{COLS}, row-major) and an int8 vector (length {ROWS}). Result is an int32 vector of length {COLS}."
description = f"""
This HLS kernel performs a matrix-vector multiplication where the matrix is a {ROWS}x{COLS} matrix of 8-bit integers (ap_int<8>), 
and the vector is a {ROWS}-element vector of 8-bit integers (ap_int<8>). The result is a {COLS}-element vector of 32-bit integers (ap_int<32>).
The matrix is stored in a row-major format, and the output vector is computed as the dot product of each row of the matrix with the input vector.
"""

inputs_info = [
    {
        "name": "matrix",  # Name of the input matrix
        "array_dimension": 2,  # Number of dimensions in the input matrix
        "shape": (ROWS, COLS),  # Shape of the input matrix
        "total_elements": ROWS * COLS,  # Total number of elements in the input matrix
        "dtype": "int8",
        "iotype": "in",  # Input matrix for the kernel
        "min": -20,  # Minimum value for the input matrix
        "max": 20,  # Maximum value for the input matrix
    },
    {
        "name": "vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (COLS,),  # Shape of the input vector
        "total_elements": COLS,  # Total number of elements in the input vector
        "dtype": "int8",
        "iotype": "in",  # Input vector for the kernel
        "min": -20,  # Minimum value for the input vector
        "max": 20,  # Maximum value for the input vector
    },
    {
        "name": "result",  # Name of the output vector
        "array_dimension": 1,  # Number of dimensions in the output vector
        "shape": (ROWS,),  # Shape of the output vector
        "total_elements": ROWS,  # Total number of elements in the output vector
        "dtype": "int32",
        "iotype": "out",  # Output vector for the kernel
    }
]

def behavioral(matrix, vector):
    mat = matrix.reshape((ROWS, COLS)).astype(np.int32)
    vec = vector.astype(np.int32)
    result = np.zeros(ROWS, dtype=np.int32)
    for i in range(ROWS):
        acc = 0
        for j in range(COLS):
            acc += mat[i, j] * vec[j]
        result[i] = acc
    return result

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")