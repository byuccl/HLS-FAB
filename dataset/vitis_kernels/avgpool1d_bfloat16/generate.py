# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_types, generate_examples

num_values = 1024  # Length of the input vector
window_size = 2
stride = 2
output_length = (num_values - window_size) // stride + 1


function_signature = "void avgpool1d_bfloat16(ap_float<16,8> *in_buffer, ap_float<16,8> *out_buffer)"

description = f"""
An HLS kernel that performs a 1D average pooling operation on an input vector with window_size and stride as runtime parameters.
    First parameter: Pointer to the input buffer of {num_values} ap_float<16,8> values.
    Second parameter: Pointer to the output buffer of ap_float<16,8> values.
The kernel computes the average of each window of size {window_size} in the input buffer,
with a stride of {stride}, and stores the result in the output buffer.
The output buffer will have a length of (input_length - window_size) // stride + 1.
The input buffer is expected to be at least {num_values} ap_float<16,8> values
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (output_length,),  # Shape of the output buffer
        "total_elements": output_length,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    output_length = (len(input_buffer) - window_size) // stride + 1
    output_buffer = np.zeros(output_length, dtype=bfloat16)
    output_idx = 0
    for i in range(0, len(input_buffer) - window_size + 1, stride):
        window = input_buffer[i:i+window_size]
        output_buffer[output_idx] = np.mean(window)
        output_idx += 1
    return output_buffer

examples = generate_examples(inputs_info)   

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")