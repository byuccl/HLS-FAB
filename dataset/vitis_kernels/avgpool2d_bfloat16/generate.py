import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
num_rows = 32
num_cols = 32
window_size = 2
stride = 2
output_rows = num_rows // stride
output_cols = num_cols // stride

function_signature = "void avgpool2d_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

description = f"""
This HLS kernel performs a 2D average pooling operation on a 32x32 input matrix of ap_float<16,8> values.
The kernel uses a 2x2 window and a stride of 2 to compute the average of each window in the input matrix.
The result is stored in the output matrix, which has dimensions of 16x16.
Parameters:
    - First parameter: Pointer to the input buffer of {num_rows}x{num_cols} ap_float<16,8> values.
    - Second parameter: Pointer to the output buffer of {output_rows}x{output_cols} ap_float<16,8> values.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 2,  # Number of dimensions in the input buffer
        "shape": (num_rows, num_cols),  # Shape of the input buffer
        "total_elements": num_rows * num_cols,  # Total number of elements in the input buffer
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
    input_matrix = input_matrix.reshape((num_rows, num_cols))
    output_matrix = np.zeros((num_rows // 2, num_cols // 2), dtype=input_matrix.dtype)
    for i in range(0, num_rows, 2):
        for j in range(0, num_cols, 2):
            window = input_matrix[i:i+2, j:j+2]
            output_matrix[i//2, j//2] = np.mean(window)
    return output_matrix.flatten().astype(bfloat16)

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")