# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

VECTOR_SIZE = 256

function_signature = "void relu_bfloat16_cast_uint8(ap_float<16,8>* input_vector, ap_uint<8>* output_vector)"

description = f"""
This HLS kernel applies the ReLU function to a bfloat16 input vector of size {VECTOR_SIZE}, where negative values are set to zero. 
The result is then rounded to the nearest integer, clamped between 0 and 255, and cast to uint8_t. 
The input is a pointer to a buffer of {VECTOR_SIZE} ap_float<16,8> values, and the output is a pointer to a buffer of {VECTOR_SIZE} ap_uint<8> values.
"""

inputs_info = [
    {
        "name": "input_vector",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "bfloat16",
        "iotype": "in",
        "min": -128,
        "max": 128,
    },
    {
        "name": "output_vector",
        "array_dimension": 1,
        "shape": (VECTOR_SIZE,),
        "total_elements": VECTOR_SIZE,
        "dtype": "uint8",
        "iotype": "out",
    }
]

def behavioral(input_vector):
    relu = np.maximum(input_vector, 0)
    rounded = np.floor(relu+(0.5))
    clipped = np.clip(rounded, 0, 255)
    return clipped.astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

tolerances = {"atol": 0.03, "rtol": 0.03}

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
    tolerances=tolerances
)

pc.write_json("kernel.json")