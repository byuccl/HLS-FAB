import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
IN_ROWS = 16
IN_COLS = 16
KERNEL_SIZE = 4
STRIDE = 2
OUT_ROWS = (IN_ROWS - KERNEL_SIZE) // STRIDE + 1
OUT_COLS = (IN_COLS - KERNEL_SIZE) // STRIDE + 1

function_signature = "void conv2d_k4_s2_bias_relu_bfloat16(ap_float<16,8>* in_buffer, ap_float<16,8>* kernel, ap_float<16,8>* out_buffer, ap_float<16,8> bias, uint32_t stride)"

description = f"""
This HLS kernel performs a 2D convolution on a 16x16 input matrix of bfloat16 values using a 4x4 kernel matrix of bfloat16 values. 
The operation includes a stride of {STRIDE}, adds a scalar bias, and applies a ReLU activation function. 
The output is a 7x7 matrix of bfloat16 values. 
Parameters:
    - First parameter: Pointer to the input buffer of {IN_ROWS}x{IN_COLS} ap_float<16,8> values.
    - Second parameter: Pointer to the kernel buffer of {KERNEL_SIZE}x{KERNEL_SIZE} ap_float<16,8> values.
    - Third parameter: Pointer to the output buffer of {OUT_ROWS}x{OUT_COLS} ap_float<16,8> values.
    - Fourth parameter: Scalar bias of type ap_float<16,8>.
    - Fifth parameter: Stride of type uint32_t.
"""

inputs_info = [
    {
        "name": "in_buffer",
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
        "name": "out_buffer",
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
    },
    {
        "name": "stride",
        "array_dimension": 0,
        "shape": (),
        "total_elements": 1,
        "dtype": "uint32_t",
        "iotype": "in",
        "min": 1,
        "max": 2,
    }
]

def behavioral(in_buffer, kernel, bias, stride):
    in_rows, in_cols = in_buffer.shape
    kernel_size = kernel.shape[0]
    out_rows = (in_rows - kernel_size) // stride + 1
    out_cols = (in_cols - kernel_size) // stride + 1
    out = np.zeros((out_rows, out_cols), dtype=bfloat16)
    for i in range(out_rows):
        for j in range(out_cols):
            window = in_buffer[i*stride:i*stride+kernel_size, j*stride:j*stride+kernel_size].astype(np.float32)
            k = kernel.astype(np.float32)
            acc = np.sum(window * k) + float(bias)
            relu = max(acc, 0.0)
            out[i, j] = bfloat16(relu)
    return out

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")