import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256
kernel_size = 3
stride = 1
output_size = (vector_size - kernel_size) // stride + 1

function_signature = "void conv1d_int32(ap_int<32> *in_buffer, ap_int<32> *kernel, ap_int<32> *out_buffer)"

description = f"""
This HLS kernel performs a 1D convolution operation on an input vector using a specified kernel and stride.
Parameters to the function include:
    - A pointer to the input buffer of {vector_size} ap_int<32> values.
    - A pointer to the kernel buffer of {kernel_size} ap_int<32> values.
    - A pointer to the output buffer of ap_int<32> values.
The kernel computes the convolution of the input buffer with the kernel, using the specified stride, and stores the result in the output buffer.
The output buffer will have a length of (input_length - kernel_size) // stride + 1.
"""

inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "int32",
        "iotype": "in",
        "min": -10,
        "max": 10,
    },
    {
        "name": "kernel",
        "array_dimension": 1,
        "shape": (kernel_size,),
        "total_elements": kernel_size,
        "dtype": "int32",
        "iotype": "in",
        "min": -10,
        "max": 10,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (output_size,),
        "total_elements": output_size,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(in_buffer, kernel):
    output_size = (in_buffer.shape[0] - kernel.shape[0]) // stride + 1
    out_buffer = np.zeros(output_size, dtype=np.int32)
    for i in range(output_size):
        conv_sum = 0
        for j in range(kernel.shape[0]):
            conv_sum += in_buffer[i * stride + j] * kernel[j]
        out_buffer[i] = conv_sum
    return out_buffer

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")