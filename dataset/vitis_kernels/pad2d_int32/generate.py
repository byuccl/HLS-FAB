import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants for input matrix dimensions and padding
input_rows = 16
input_cols = 16
pad_size = 2
pad_value = 0

# Function signature for the HLS kernel
function_signature = "void pad2d_int32(ap_int<32> *input_matrix, ap_int<32> *padded_matrix, uint32_t pad_size, ap_int<32> pad_value)"

# Detailed description of the kernel
description = f"""
This HLS kernel performs a 2D padding operation on an input matrix of 32-bit integers.
Parameters to the function include:
    - A pointer to the input matrix of dimensions {input_rows}x{input_cols}.
    - A pointer to the output padded matrix.
    - An integer pad_size indicating the number of padding layers to add around the input matrix.
    - An integer pad_value which is used to fill the padding layers.
The output matrix will have dimensions ({input_rows} + 2 * pad_size) x ({input_cols} + 2 * pad_size).
"""

# Information about the inputs and outputs
inputs_info = [
    {
        "name": "input_matrix",
        "array_dimension": 2,
        "shape": (input_rows, input_cols),
        "total_elements": input_rows * input_cols,
        "dtype": "int32",
        "iotype": "in",
        "min": -10,
        "max": 10,
    },
    {
        "name": "padded_matrix",
        "array_dimension": 2,
        "shape": (input_rows + 2 * pad_size, input_cols + 2 * pad_size),
        "total_elements": (input_rows + 2 * pad_size) * (input_cols + 2 * pad_size),
        "dtype": "int32",
        "iotype": "out",
    }
]

# Behavioral function to simulate the kernel's operation
def behavioral(input_matrix, pad_size=pad_size, pad_value=pad_value):  
    n_rows = int(np.sqrt(len(input_matrix)))  # This assumes a square matrix
    n_cols = n_rows
    
    # Reshape to original 2D form
    input_matrix = input_matrix.reshape(n_rows, n_cols)
    
    # Now proceed with padding as before
    input_rows, input_cols = input_matrix.shape
    padded_rows = input_rows + 2 * pad_size
    padded_cols = input_cols + 2 * pad_size
    padded_matrix = np.full((padded_rows, padded_cols), pad_value, dtype=input_matrix.dtype)
    padded_matrix[pad_size:pad_size + input_rows, pad_size:pad_size + input_cols] = input_matrix
    return padded_matrix

# Generate examples for testing
examples = generate_examples(inputs_info=inputs_info)

# Create the prompt constructor
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")