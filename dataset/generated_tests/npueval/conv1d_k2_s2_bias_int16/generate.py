import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 128
KERNEL_SIZE = 2
STRIDE = 2
OUTPUT_SIZE = (VECTOR_SIZE - KERNEL_SIZE) // STRIDE + 1

function_signature = "void conv1d_k2_s2_bias_int16(ap_int<16> *in_buffer, ap_int<16> *kernel, ap_int<16> *out_buffer, ap_int<16> bias)"

description = f"""
This HLS kernel performs a 1D convolution on an input vector of 16-bit integers with a kernel size of {KERNEL_SIZE} and a stride of {STRIDE}.
The kernel also adds a scalar 16-bit integer bias to the accumulator before writing the result to the output buffer.
Parameters:
    - in_buffer: Pointer to the input buffer containing {VECTOR_SIZE} ap_int<16> elements.
    - kernel: Pointer to the kernel buffer containing {KERNEL_SIZE} ap_int<16> elements.
    - out_buffer: Pointer to the output buffer which will contain {OUTPUT_SIZE} ap_int<16> elements.
    - bias: A scalar ap_int<16> value added to each convolution result.
The output is saturated to the range of 16-bit integers.
"""

inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "int16",
        "iotype": "in",
        "min": -100,
        "max": 100,
    },
    {
        "name": "kernel",
        "array_dimension": 1,
        "shape": (KERNEL_SIZE,),
        "total_elements": KERNEL_SIZE,
        "dtype": "int16",
        "iotype": "in",
        "min": -10,
        "max": 10,
    },
    {
        "name": "bias",
        "array_dimension": 0,
        "shape": (),
        "total_elements": 1,
        "dtype": "int16",
        "iotype": "in",
        "min": -10,
        "max": 10,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (OUTPUT_SIZE,),
        "total_elements": OUTPUT_SIZE,
        "dtype": "int16",
        "iotype": "out",
    }
]

def behavioral(in_buffer, kernel, bias):
    num_windows = (VECTOR_SIZE - KERNEL_SIZE) // STRIDE + 1
    out = np.zeros(num_windows, dtype=np.int16)
    for i in range(num_windows):
        acc = int(bias)
        for j in range(KERNEL_SIZE):
            acc += int(in_buffer[i * STRIDE + j]) * int(kernel[j])
        acc = np.clip(acc, -32768, 32767)
        out[i] = acc
    return out

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")