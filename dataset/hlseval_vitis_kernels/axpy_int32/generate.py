# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 4096

function_signature = "void axpy_int32(ap_int<32> *x_vector, ap_int<32> alpha, ap_int<32> *y_vector)"

description = f"""
This HLS kernel implements the AXPY operation on two input vectors of 32-bit integers. 
The operation is defined as Y = alpha * X + Y, where X and Y are vectors and alpha is a scalar.
Parameters to the function include two ap_int<32> pointers and one ap_int<32> scalar:
    The first pointer points to an array containing {num_values} int32 values. This is the input vector X.
    The second parameter is an int32 scalar value, alpha.
    The third pointer points to an array containing {num_values} int32 values. This is the input/output vector Y.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

inputs_info = [
    {
        "name": "x_vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (num_values,),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": -32000,  # Minimum value for the input vector
        "max": 32000,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "alpha",  # Name of the input value
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": -32000,  # Minimum value for the input vector
        "max": 32000,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "y_vector",  # Name of the result
        "array_dimension": 1,  # Number of dimensions in the result
        "shape": (num_values,),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "int32",
        "iotype": "inout",  # Output for the kernel
        "min": -32000,  # Minimum value for the input vector
        "max": 32000,  # Maximum value for the input vector
    }  # Output for the kernel

]

def behavioral(input_x_vector, input_alpha, input_y_vector):
    return input_alpha * input_x_vector + input_y_vector

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")