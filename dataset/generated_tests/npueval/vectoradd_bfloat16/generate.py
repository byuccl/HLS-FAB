# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vectors

function_signature = "void vectoradd_bfloat16(ap_float<16,8> *in0, ap_float<16,8> *in1, ap_float<16,8> *out)"

short_description = "This HLS kernel computes the elementwise addition of two bfloat16 input vectors of size 256, writing the result to an output bfloat16 vector."
description = f"""
This HLS kernel computes the elementwise addition of two input vectors of bfloat16 values.
Parameters to the function include three ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as the first input vector.
    The second pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as the second input vector.
    The third pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be modified and viewed as the output vector.
The kernel performs elementwise addition of the two input vectors and stores the result in the output vector.
"""

inputs_info = [
    {
        "name": "in0",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "in1",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "out",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(in0, in1):
    return (in0 + in1).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")