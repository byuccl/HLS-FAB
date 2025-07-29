import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
input_rows = 32
input_cols = 32
window_size = 2
stride = 2
output_rows = input_rows // stride
output_cols = input_cols // stride

function_signature = "void maxpool2d_relu_bfloat16(ap_float<16,8>* input, ap_float<16,8>* output)"

short_description = (
    "A kernel that performs 2D max pooling and applies ReLU on an input bfloat16 matrix (32x32), "
    "using a 2x2 window and stride of 2. The output is a 16x16 matrix in the output buffer where "
    "each element is the maximum of each 2x2 patch with ReLU applied."
)
description = f"""
This HLS kernel performs a 2D max pooling operation followed by a ReLU activation on an input matrix of bfloat16 values.
The input is a {input_rows}x{input_cols} matrix, and the operation uses a {window_size}x{window_size} window with a stride of {stride}.
The output is a {output_rows}x{output_cols} matrix where each element is the maximum value of each {window_size}x{window_size} patch from the input,
with ReLU applied to ensure non-negative outputs.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 2,  # Number of dimensions in the input buffer
        "shape": (input_rows, input_cols),  # Shape of the input buffer
        "total_elements": input_rows * input_cols,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3.0,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 2,  # Number of dimensions in the output buffer
        "shape": (output_rows, output_cols),  # Shape of the output buffer
        "total_elements": output_rows * output_cols,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_matrix):
    input_matrix = input_matrix.reshape((input_rows, input_cols))
    out_rows = input_rows // 2
    out_cols = input_cols // 2
    output_matrix = np.zeros((out_rows, out_cols), dtype=bfloat16)
    for i in range(0, input_rows, 2):
        for j in range(0, input_cols, 2):
            window = input_matrix[i:i+2, j:j+2]
            maxval = np.max(window).astype(np.float32)
            relu = max(maxval, 0.0)
            output_matrix[i//2, j//2] = bfloat16(relu)
    return output_matrix.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")