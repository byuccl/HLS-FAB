# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
num_values = 1024

function_signature = "void inverse_uint8(ap_uint<8> *in_buffer, ap_uint<8>* out_buffer)"

description = f"""
This HLS kernel inverts all input values of a buffer containing a uint8 array.
Parameters to the function include two ap_uint<8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_uint<8> values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of ap_uint<8> values to be modified and viewed as the output to the function.
The kernel computes the inverse of each element in the input buffer by subtracting the element from 255 and stores the result in the output buffer.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "uint8",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0,  # Minimum value for the input buffer
        "max": 255,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "uint8",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    return (255 - input_buffer).astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")