import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
ROWS = 32
COLS = 32
WINDOW_SIZE = 2
STRIDE = 2
OUTPUT_ROWS = ROWS // STRIDE
OUTPUT_COLS = COLS // STRIDE

function_signature = "void maxpool2d_bfloat16(ap_float<16,8>* input, ap_float<16,8>* output)"

description = f"""
This HLS kernel performs a 2D max pooling operation on a 32x32 input matrix of ap_float<16,8> values.
The kernel uses a 2x2 window and a stride of 2 to compute the maximum value within each window.
The result is stored in the output matrix, which has dimensions of {OUTPUT_ROWS}x{OUTPUT_COLS}.
Parameters:
    - First parameter: Pointer to the input buffer of {ROWS}x{COLS} ap_float<16,8> values.
    - Second parameter: Pointer to the output buffer of {OUTPUT_ROWS}x{OUTPUT_COLS} ap_float<16,8> values.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 2,  # Number of dimensions in the input buffer
        "shape": (ROWS, COLS),  # Shape of the input buffer
        "total_elements": ROWS * COLS,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3.0,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 2,  # Number of dimensions in the output buffer
        "shape": (OUTPUT_ROWS, OUTPUT_COLS),  # Shape of the output buffer
        "total_elements": OUTPUT_ROWS * OUTPUT_COLS,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_matrix):
    rows, cols = input_matrix.shape
    output_matrix = np.zeros((rows // 2, cols // 2), dtype=input_matrix.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input_matrix[i:i+2, j:j+2]
            output_matrix[i//2, j//2] = np.max(window)
    return output_matrix

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")