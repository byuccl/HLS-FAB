# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants for vector and matrix dimensions
vector_size = 16
matrix_cols = 16

function_signature = "void vectormatrix_mult_int32(ap_int<32> *vector, ap_int<32> *matrix, ap_int<32> *result)"

description = f"""
This HLS kernel performs a vector-matrix multiplication. 
The function takes three parameters:
    1. A pointer to a vector of {vector_size} ap_int<32> values.
    2. A pointer to a matrix of shape ({vector_size}, {matrix_cols}) represented as a flat array of ap_int<32> values.
    3. A pointer to a result vector of {matrix_cols} ap_int<32> values.
The kernel computes the product of the input vector and matrix, storing the result in the output vector.
"""

inputs_info = [
    {
        "name": "vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (vector_size,),  # Shape of the input vector
        "total_elements": vector_size,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": -10,  # Minimum value for the input vector
        "max": 10,  # Maximum value for the input vector
    },
    {
        "name": "matrix",  # Name of the input matrix
        "array_dimension": 2,  # Number of dimensions in the input matrix
        "shape": (vector_size, matrix_cols),  # Shape of the input matrix
        "total_elements": vector_size * matrix_cols,  # Total number of elements in the input matrix
        "dtype": "int32",
        "iotype": "in",  # Input matrix for the kernel
        "min": -10,  # Minimum value for the input matrix
        "max": 10,  # Maximum value for the input matrix
    },
    {
        "name": "result",  # Name of the output vector
        "array_dimension": 1,  # Number of dimensions in the output vector
        "shape": (matrix_cols,),  # Shape of the output vector
        "total_elements": matrix_cols,  # Total number of elements in the output vector
        "dtype": "int32",
        "iotype": "out",  # Output vector for the kernel
    }
]

def behavioral(vector, matrix):
    m = matrix.reshape((vector_size, matrix_cols))
    result = np.zeros(matrix_cols, dtype=np.int32)
    for i in range(matrix_cols):
        for j in range(vector_size):
            result[i] += vector[j] * m[j, i]
    return result

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")