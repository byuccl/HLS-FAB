import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants for the input and output dimensions
input_rows = 32
input_cols = 32
window_size = 2
stride = 2
output_rows = input_rows // stride
output_cols = input_cols // stride

function_signature = "void avgpool2d_relu_bfloat16(ap_float<16,8>* input, ap_float<16,8>* output)"

description = f"""
This HLS kernel performs a 2D average pooling operation followed by a ReLU activation on an input matrix of size {input_rows}x{input_cols}.
The operation uses a window of size {window_size}x{window_size} and a stride of {stride}.
The output is a matrix of size {output_rows}x{output_cols}, where each element is the result of applying the average pooling and ReLU activation to the corresponding window in the input matrix.
All operations are performed using the bfloat16 data type.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 2,  # Number of dimensions in the input buffer
        "shape": (input_rows, input_cols),  # Shape of the input buffer
        "total_elements": input_rows * input_cols,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -4.0,  # Minimum value for the input buffer
        "max": 4.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 2,  # Number of dimensions in the output buffer
        "shape": (output_rows, output_cols),  # Shape of the output buffer
        "total_elements": output_rows * output_cols,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_matrix):
    rows, cols = input_matrix.shape
    out_rows = rows // stride
    out_cols = cols // stride
    output_matrix = np.zeros((out_rows, out_cols), dtype=bfloat16)
    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            window = input_matrix[i:i+window_size, j:j+window_size].astype(np.float32)
            avg = window.mean()
            relu = max(avg, 0.0)
            output_matrix[i//stride, j//stride] = bfloat16(relu)
    return output_matrix

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")