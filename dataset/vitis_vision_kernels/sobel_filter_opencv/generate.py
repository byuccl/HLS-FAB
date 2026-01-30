# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples
import cv2

# Image dimensions
rows = 128
cols = 128
num_values = rows * cols

function_signature = "void sobel_filter_opencv(ap_uint<8>* img_inp, ap_uint<8>* img_out1, ap_uint<8>* img_out2)"

description = f"""
This HLS kernel implements a Sobel filter to compute the gradients of an input image in both x and y directions. 
The input image is 128x128 pixels, and each pixel is represented by an 8-bit unsigned integer.
Parameters to the function include three uint8_t pointers:
    The first pointer points to the start of a {rows * cols}-element array representing the input image.
    The second and third pointers point to the start of {rows * cols}-element arrays representing the output gradient images in the x and y directions, respectively.
The border pixels are set to a constant 0.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

inputs_info = [
    {
        "name": "img_inp",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (rows,cols),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "img_out1",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (rows,cols),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "out",  # Input vector for the kernel
    },  # Input vector for the kernel
    {
        "name": "img_out2",  # Name of the result
        "array_dimension": 2,  # Number of dimensions in the result
        "shape": (rows,cols),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint16",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]


def behavioral(input_image):
    input_image_2d = input_image.reshape((rows, cols))
    grad_x = cv2.Sobel(input_image_2d, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input_image_2d, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return abs_grad_x.flatten(), abs_grad_y.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")