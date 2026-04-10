import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
input_rows = 16
input_cols = 16
kernel_size = 3
stride = 1
output_rows = (input_rows - kernel_size) // stride + 1
output_cols = (input_cols - kernel_size) // stride + 1

function_signature = "void conv2d_int32(int32_t *input_matrix, int32_t* kernel, int32_t* output_matrix)"

description = f"""
This HLS kernel performs a 2D convolution operation on an input matrix using a specified kernel and stride.
Parameters to the function include:
    - A pointer to the input matrix of size {input_rows}x{input_cols} with int32_t values.
    - A pointer to the kernel matrix of size {kernel_size}x{kernel_size} with int32_t values.
    - A pointer to the output matrix which will store the result of the convolution operation.
    - A stride value that determines the step size for the convolution operation.
The output matrix will have dimensions of {output_rows}x{output_cols}.
"""

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
        "name": "kernel",
        "array_dimension": 2,
        "shape": (kernel_size, kernel_size),
        "total_elements": kernel_size * kernel_size,
        "dtype": "int32",
        "iotype": "in",
        "min": -10,
        "max": 10,
    },
    {
        "name": "output_matrix",
        "array_dimension": 2,
        "shape": (output_rows, output_cols),
        "total_elements": output_rows * output_cols,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(input_matrix, kernel):
    input_matrix = input_matrix.reshape((input_rows, input_cols))
    kernel = kernel.reshape((kernel_size, kernel_size))
    output_matrix = np.zeros((output_rows, output_cols), dtype=np.int32)
    for i in range(output_rows):
        for j in range(output_cols):
            conv_sum = 0
            for ki in range(kernel_size):
                for kj in range(kernel_size):
                    conv_sum += input_matrix[i * stride + ki, j * stride + kj] * kernel[ki, kj]
            output_matrix[i, j] = conv_sum
    return output_matrix.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")