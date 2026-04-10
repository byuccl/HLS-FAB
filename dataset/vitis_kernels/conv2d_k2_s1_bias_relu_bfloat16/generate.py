import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
IN_ROWS, IN_COLS = 16, 16
KERNEL_SIZE = 2
STRIDE = 1
OUT_ROWS = (IN_ROWS - KERNEL_SIZE) // STRIDE + 1
OUT_COLS = (IN_COLS - KERNEL_SIZE) // STRIDE + 1

function_signature = "void conv2d_k2_s1_bias_relu_bfloat16(ap_float<16,8>* input, ap_float<16,8>* kernel, ap_float<16,8>* output, ap_float<16,8> bias)"

description = f"""
This HLS kernel performs a 2D convolution operation on a 16x16 input matrix using a 2x2 kernel, both of type ap_float<16,8>. 
The operation includes a stride of 1, adds a bias of type ap_float<16,8>, and applies a ReLU activation function to each output element. 
The result is stored in a 15x15 output matrix of type ap_float<16,8>.
Parameters:
    - First parameter: Pointer to the input buffer of {IN_ROWS}x{IN_COLS} ap_float<16,8> values.
    - Second parameter: Pointer to the kernel buffer of {KERNEL_SIZE}x{KERNEL_SIZE} ap_float<16,8> values.
    - Third parameter: Pointer to the output buffer of {OUT_ROWS}x{OUT_COLS} ap_float<16,8> values.
    - Fourth parameter: A single ap_float<16,8> value representing the bias.
"""

inputs_info = [
    {
        "name": "input",
        "array_dimension": 2,
        "shape": (IN_ROWS, IN_COLS),
        "total_elements": IN_ROWS * IN_COLS,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "kernel",
        "array_dimension": 2,
        "shape": (KERNEL_SIZE, KERNEL_SIZE),
        "total_elements": KERNEL_SIZE * KERNEL_SIZE,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -1.0,
        "max": 1.0,
    },
    {
        "name": "output",
        "array_dimension": 2,
        "shape": (OUT_ROWS, OUT_COLS),
        "total_elements": OUT_ROWS * OUT_COLS,
        "dtype": "bfloat16",
        "iotype": "out",
    },
    {
        "name": "bias",
        "array_dimension": 0,
        "shape": (),
        "total_elements": 1,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -1.0,
        "max": 1.0,
    }
]

def behavioral(input, kernel, bias):
    input = input.reshape((IN_ROWS, IN_COLS))
    kernel = kernel.reshape((KERNEL_SIZE, KERNEL_SIZE))
    output = np.zeros((OUT_ROWS, OUT_COLS), dtype=bfloat16)
    for i in range(OUT_ROWS):
        for j in range(OUT_COLS):
            acc = float(bias)
            for ki in range(KERNEL_SIZE):
                for kj in range(KERNEL_SIZE):
                    acc += float(input[i + ki, j + kj]) * float(kernel[ki, kj])
            relu = max(acc, 0.0)
            output[i, j] = bfloat16(relu)
    return output.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")