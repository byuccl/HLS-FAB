import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 256
KERNEL_SIZE = 2
STRIDE = 1
OUTPUT_SIZE = (VECTOR_SIZE - KERNEL_SIZE) // STRIDE + 1

function_signature = "void conv1d_k2_s1_bias_relu_bfloat16(ap_float<16,8>* in_buffer, ap_float<16,8>* kernel, ap_float<16,8>* out_buffer, ap_float<16,8> bias)"

description = f"""
This HLS kernel performs a 1D convolution operation on an input vector of bfloat16 values with a kernel size of {KERNEL_SIZE} and a stride of {STRIDE}.
The kernel also adds a bfloat16 bias to the result of the convolution and applies a ReLU activation function.
The input buffer is expected to be {VECTOR_SIZE} elements long, and the kernel buffer is {KERNEL_SIZE} elements long.
The output buffer will contain the results of the convolution, bias addition, and ReLU activation, with a length of {OUTPUT_SIZE}.
"""

inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "kernel",
        "array_dimension": 1,
        "shape": (KERNEL_SIZE,),
        "total_elements": KERNEL_SIZE,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -1.0,
        "max": 1.0,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (OUTPUT_SIZE,),
        "total_elements": OUTPUT_SIZE,
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

def behavioral(in_buffer, kernel, bias):
    in_buffer = in_buffer.astype(np.float32)
    kernel = kernel.astype(np.float32)
    bias = float(bias)
    output_size = (in_buffer.shape[0] - kernel.shape[0]) // STRIDE + 1
    out = np.zeros(output_size, dtype=bfloat16)
    for i in range(output_size):
        acc = 0.0
        for j in range(kernel.shape[0]):
            acc += in_buffer[i * STRIDE + j] * kernel[j]
        acc += bias
        acc = acc if acc > 0.0 else 0.0
        out[i] = bfloat16(acc)
    return out

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")