# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
VECTOR_SIZE = 256

function_signature = "void elementwise_max_bfloat16(ap_float<16,8> *input0, ap_float<16,8> *input1, ap_float<16,8> *output)"

description = f"""
This HLS kernel performs an elementwise maximum operation on two input vectors of bfloat16 values.
Parameters to the function include three ap_float<16,8> pointers:
    The first pointer points to the start of a {VECTOR_SIZE}-element array of ap_float<16,8> values to be used as the first input vector.
    The second pointer points to the start of a {VECTOR_SIZE}-element array of ap_float<16,8> values to be used as the second input vector.
    The third pointer points to the start of a {VECTOR_SIZE}-element array of ap_float<16,8> values where the elementwise maximum of the two input vectors will be stored as output.
"""

inputs_info = [
    {
        "name": "input0",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (VECTOR_SIZE,),  # Shape of the input buffer
        "total_elements": VECTOR_SIZE,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "input1",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (VECTOR_SIZE,),  # Shape of the input buffer
        "total_elements": VECTOR_SIZE,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (VECTOR_SIZE,),  # Shape of the output buffer
        "total_elements": VECTOR_SIZE,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(input0, input1):
    return np.maximum(input0, input1).astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")