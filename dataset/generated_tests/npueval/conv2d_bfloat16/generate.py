import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
input_rows = 16
input_cols = 16
kernel_size = 2
stride = 1
output_rows = (input_rows - kernel_size) // stride + 1
output_cols = (input_cols - kernel_size) // stride + 1

function_signature = "void conv2d_bfloat16(ap_float<16,8> *input_matrix, ap_float<16,8> *kernel, ap_float<16,8> *output_matrix)"

description = f"""
This HLS kernel performs a 2D convolution on a bfloat16 input matrix with a bfloat16 kernel and a runtime stride parameter.
    First parameter: Pointer to the input matrix of {input_rows}x{input_cols} ap_float<16,8> values.
    Second parameter: Pointer to the kernel matrix of {kernel_size}x{kernel_size} ap_float<16,8> values.
    Third parameter: Pointer to the output matrix of ap_float<16,8> values.
The kernel computes the convolution of the input matrix with the kernel matrix using a stride of {stride}, and stores the result in the output matrix.
The output matrix will have dimensions of {output_rows}x{output_cols}.
"""

inputs_info = [
    {
        "name": "input_matrix",
        "array_dimension": 2,
        "shape": (input_rows, input_cols),
        "total_elements": input_rows * input_cols,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "kernel",
        "array_dimension": 2,
        "shape": (kernel_size, kernel_size),
        "total_elements": kernel_size * kernel_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "output_matrix",
        "array_dimension": 2,
        "shape": (output_rows, output_cols),
        "total_elements": output_rows * output_cols,
        "dtype": "bfloat16",
        "iotype": "out",
    }
]

def behavioral(input_matrix, kernel):
    input_matrix = input_matrix.reshape((input_rows, input_cols))
    kernel = kernel.reshape((kernel_size, kernel_size))
    output_rows = (input_rows - kernel_size) // stride + 1
    output_cols = (input_cols - kernel_size) // stride + 1

    output_matrix = np.zeros((output_rows, output_cols), dtype=np.float32)

    for i in range(output_rows):
        for j in range(output_cols):
            region = input_matrix[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size].astype(np.float32)
            kv = kernel.astype(np.float32)
            output_matrix[i, j] = np.sum(region * kv)
    return output_matrix.astype(bfloat16).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")