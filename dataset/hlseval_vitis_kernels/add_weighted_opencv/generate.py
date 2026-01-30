# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
num_values = height * width

function_signature = "void add_weighted_opencv(ap_uint<8> *src1, ap_uint<8> *src2, ap_uint<8> *dst, float alpha, float beta, float gamma)"


description = f"""
This HLS kernel implements a function to calculate a weighted sum of two input images and generates the result in the output image.
Parameters to the function include two ap_uint<8> pointers for the input images, one ap_uint<8> pointer for the output image, and three float values for the weights:
    The first pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the first input image.
    The second pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the second input image.
    The third pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the output image.
    The fourth parameter is a float value representing the weight alpha.
    The fifth parameter is a float value representing the weight beta.
    The sixth parameter is a float value representing the weight gamma.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

inputs_info = [
    {
        "name": "src1",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "src2",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "result",  # Name of the result
        "array_dimension": 2,  # Number of dimensions in the result
        "shape": (height,width),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint16",
        "iotype": "out",  # Output for the kernel
    },  # Output for the kernel
    {
        "name": "alpha",  # Name of the input value
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "float32",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 1,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "beta",  # Name of the input value
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "float32",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 1,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "gamma",  # Name of the input value
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "float32",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 1,  # Maximum value for the input vector
    },  # Input vector for the kernel
]

def behavioral(input_image1, input_image2, alpha, beta, gamma):
    weighted_sum = alpha * input_image1 + beta * input_image2 + gamma
    return np.clip(weighted_sum, 0, 255).astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")