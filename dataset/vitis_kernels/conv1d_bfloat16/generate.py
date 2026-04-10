import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
vector_size = 256
kernel_size = 2
stride = 1
output_size = (vector_size - kernel_size) // stride + 1

function_signature = "void conv1d_bfloat16(ap_float<16,8> *in_buffer, ap_float<16,8> *kernel, ap_float<16,8> *out_buffer)"

description = f"""
This HLS kernel performs a 1D convolution operation on a bfloat16 input vector using a bfloat16 kernel with a specified stride.
Parameters to the function include:
    - A pointer to the input buffer of {vector_size} ap_float<16,8> values.
    - A pointer to the kernel buffer of {kernel_size} ap_float<16,8> values.
    - A pointer to the output buffer where the result of the convolution will be stored.
The output buffer will have a length of (input_length - kernel_size) // stride + 1.
"""

inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (vector_size,),
        "total_elements": vector_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "kernel",
        "array_dimension": 1,
        "shape": (kernel_size,),
        "total_elements": kernel_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -1.0,
        "max": 1.0,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (output_size,),
        "total_elements": output_size,
        "dtype": "bfloat16",
        "iotype": "out",
    }
]

def behavioral(in_buffer, kernel):
    output_size = (in_buffer.shape[0] - kernel.shape[0]) // stride + 1
    out_buffer = np.zeros(output_size, dtype=bfloat16)
    for i in range(output_size):
        acc = 0.0
        for j in range(kernel.shape[0]):
            acc += float(in_buffer[i * stride + j]) * float(kernel[j])
        out_buffer[i] = bfloat16(acc)
    return out_buffer

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")