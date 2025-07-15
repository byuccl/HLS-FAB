import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants for the input and output dimensions
input_rows = 16
input_cols = 16
window_size = 2
stride = 2
output_rows = input_rows // stride
output_cols = input_cols // stride

function_signature = "void maxpool2d_relu_int8(ap_int<8> *input, ap_int<8> *output)"

description = f"""
This HLS kernel performs a 2D max pooling operation followed by a ReLU activation on an input tensor of 8-bit integers.
The kernel uses a {window_size}x{window_size} window with a stride of {stride} on a {input_rows}x{input_cols} input tensor.
After max pooling, the ReLU activation is applied, setting all negative values to zero.
The result is stored in an {output_rows}x{output_cols} output tensor.
Parameters:
    - First parameter: Pointer to the input buffer of {input_rows}x{input_cols} ap_int<8> values.
    - Second parameter: Pointer to the output buffer of {output_rows}x{output_cols} ap_int<8> values.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 2,  # Number of dimensions in the input buffer
        "shape": (input_rows, input_cols),  # Shape of the input buffer
        "total_elements": input_rows * input_cols,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -128,  # Minimum value for the input buffer
        "max": 127,  # Maximum value for the input buffer
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 2,  # Number of dimensions in the output buffer
        "shape": (output_rows, output_cols),  # Shape of the output buffer
        "total_elements": output_rows * output_cols,  # Total number of elements in the output buffer
        "dtype": "int8",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input_tensor):
    rows, cols = input_tensor.shape
    pooled = np.zeros((rows // 2, cols // 2), dtype=input_tensor.dtype)
    for i in range(0, rows, 2):
        for j in range(0, cols, 2):
            window = input_tensor[i:i+2, j:j+2]
            max_val = np.max(window)
            pooled[i//2, j//2] = max(max_val, 0)
    return pooled

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")