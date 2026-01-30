# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the image
height = 128
width = 128
num_values = height * width

function_signature = "void reduce_avg_opencv(ap_uint<8> *in_img, int dimension, ap_uint<8> *out_img)"

description = f"""
This HLS kernel implements a function to reduce a 128x128 image to a vector by calculating the mean of its rows or columns.
Parameters to the function include two ap_uint<8> pointers and an integer:
    The first pointer points to the start of a {height}x{width} array of ap_uint<8> values representing the input image.
    The second parameter is an integer representing the dimension along which the reduction is performed (0 for rows, 1 for columns).
    The third pointer points to the start of an array of ap_uint<8> values representing the output vector.
"""

source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

inputs_info = [
    {
        "name": "in_img",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "dimension",  # Name of the input value
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": 1,  # Minimum value for the input vector
        "max": 2,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "out_img",  # Name of the result
        "array_dimension": 2,  # Number of dimensions in the result
        "shape": (height,width),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint16",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

def behavioral(input_image, dimension):
    input_image = input_image.reshape((height, width))
    if dimension == 0:
        return np.mean(input_image, axis=0).astype(np.uint8)
    else:
        return np.mean(input_image, axis=1).astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")