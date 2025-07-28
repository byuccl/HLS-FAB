import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

VECTOR_SIZE = 256
KERNEL_SIZE = 4
STRIDE = 1
OUTPUT_SIZE = (VECTOR_SIZE - KERNEL_SIZE) // STRIDE + 1

function_signature = "void conv1d_k4_s1_bias_relu_bfloat16(ap_float<16,8>* in_buffer, ap_float<16,8>* kernel, ap_float<16,8>* out_buffer, ap_float<16,8> bias)"

description = f"""
This HLS kernel performs a 1D convolution on a bfloat16 input vector of length {VECTOR_SIZE} using a convolution kernel of size {KERNEL_SIZE} and a stride of {STRIDE}. 
After the convolution, a bfloat16 bias is added to each result, followed by the application of a ReLU activation function. 
The output is a bfloat16 vector of length {OUTPUT_SIZE}.
Parameters:
    - in_buffer: Pointer to the input buffer of {VECTOR_SIZE} ap_float<16,8> values.
    - kernel: Pointer to the convolution kernel of {KERNEL_SIZE} ap_float<16,8> values.
    - out_buffer: Pointer to the output buffer of {OUTPUT_SIZE} ap_float<16,8> values.
    - bias: A single ap_float<16,8> value to be added to each convolution result before applying ReLU.
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
    num_windows = (VECTOR_SIZE - KERNEL_SIZE) // STRIDE + 1
    output = np.zeros(num_windows, dtype=bfloat16)
    for i in range(num_windows):
        acc = 0.0
        for j in range(KERNEL_SIZE):
            acc += float(in_buffer[i * STRIDE + j]) * float(kernel[j])
        acc += float(bias)
        out_val = max(acc, 0.0)
        output[i] = bfloat16(out_val)
    return output

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")