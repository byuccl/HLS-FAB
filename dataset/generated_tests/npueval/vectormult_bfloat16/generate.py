# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

vector_size = 64  # Number of elements in each input vector

function_signature = "void vectormult_bfloat16(ap_float<16,8> *in_buffer1, ap_float<16,8> *in_buffer2, ap_float<16,8> *out_buffer, int32_t vector_size)"

short_description = "This is the implementation of an HLS kernel that performs an elementwise vector multiplication of 2 input vectors and stores the result in 1 output vector."

description = f"""
This HLS kernel performs an elementwise vector multiplication of two input vectors, each containing {vector_size} elements of type ap_float<16,8>.
The result of the multiplication is stored in an output vector of the same size.
Parameters:
    - First parameter: Pointer to the first input buffer of {vector_size} ap_float<16,8> values.
    - Second parameter: Pointer to the second input buffer of {vector_size} ap_float<16,8> values.
    - Third parameter: Pointer to the output buffer where the result of the elementwise multiplication will be stored.
    - Fourth parameter: Integer specifying the size of the vectors to be multiplied.
"""

inputs_info = [
    {
        "name": "in_buffer1",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "in_buffer2",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),  # Shape of the output buffer
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    },
    {
        "name": "vector_size",  
        "array_dimension": 0,  
        "shape": (),  
        "total_elements": 1,  
        "dtype": "int32",
        "iotype": "in",  
        "min": 64, 
        "max": 65,  
    },
]

def behavioral(in_buffer1, in_buffer2, vector_size):
    return (in_buffer1[:vector_size] * in_buffer2[:vector_size]).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")