# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# Constants
src_buffer_size = 256

function_signature = "void gather_bfloat16_int32idx(ap_float<16,8>* input_data, ap_int<32>* indices, ap_float<16,8>* output_data)"

description = f"""
This HLS kernel implements a gather operation using bfloat16 data type for input and output buffers, and int32 for indices.
Parameters to the function include:
    - A pointer to the input buffer of {src_buffer_size} ap_float<16,8> values.
    - A pointer to the indices buffer of {src_buffer_size} ap_int<32> values.
    - A pointer to the output buffer of {src_buffer_size} ap_float<16,8> values.
For each index in the indices buffer, the kernel performs the operation: output_data[i] = input_data[indices[i]].
The input and output buffers are expected to be at least {src_buffer_size} elements long.
"""

inputs_info = [
    {
        "name": "input_data",
        "array_dimension": 1,
        "shape": (src_buffer_size,),
        "total_elements": src_buffer_size,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -3.0,
        "max": 3.0,
    },
    {
        "name": "indices",
        "array_dimension": 1,
        "shape": (src_buffer_size,),
        "total_elements": src_buffer_size,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": src_buffer_size,
    },
    {
        "name": "output_data",
        "array_dimension": 1,
        "shape": (src_buffer_size,),
        "total_elements": src_buffer_size,
        "dtype": "bfloat16",
        "iotype": "out",
    }
]

def behavioral(input_data, indices):
    return input_data[indices]

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")