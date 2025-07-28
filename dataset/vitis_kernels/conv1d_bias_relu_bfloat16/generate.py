import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 256
KERNEL_SIZE = 4
STRIDE = 2
OUTPUT_SIZE = (VECTOR_SIZE - KERNEL_SIZE) // STRIDE + 1

function_signature = "void conv1d_bias_relu_bfloat16(ap_float<16,8>* in_buffer, ap_float<16,8>* kernel, ap_float<16,8>* out_buffer, ap_float<16,8> bias)"

description = f"""
This HLS kernel performs a 1D convolution operation on a {VECTOR_SIZE}-length input vector of bfloat16 values using a {KERNEL_SIZE}-element kernel, a bfloat16 bias, and a specified stride.
The kernel computes the convolution sum, adds the bias, applies the ReLU activation function, and writes the result to the output buffer.
The output buffer will have a length of (input_length - kernel_size) // stride + 1.
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
        "min": 0.0,
        "max": 2.0,
    }
]

def behavioral(in_buffer, kernel, bias):
    output_size = (len(in_buffer) - len(kernel)) // STRIDE + 1
    out_buffer = np.zeros(output_size, dtype=bfloat16)
    for i in range(output_size):
        acc = bfloat16(0.0)
        for j in range(len(kernel)):
            prod = bfloat16(float(in_buffer[i * STRIDE + j]) * float(kernel[j]))
            acc = bfloat16(float(acc) + float(prod))
        acc = bfloat16(float(acc) + float(bias))
        if float(acc) < 0.0:
            acc = bfloat16(0.0)
        out_buffer[i] = acc
    return out_buffer

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")