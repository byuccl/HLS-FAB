# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples
import cv2

# Set the dimensions of the image
height = 512
width = 512
num_values = height * width

function_signature = "void rotate_opencv(ap_uint<8>* img_in, ap_uint<8>* img_out, int direction)"

description = f"""
This HLS kernel implements a function to rotate a {height}x{width} image clockwise by 90, 180, or 270 degrees.
Parameters to the function include two ap_uint<8> pointers and an integer:
    The first pointer points to the start of a {height}x{width} array of ap_uint<8> values representing the input image.
    The second pointer points to the start of a {height}x{width} array of ap_uint<8> values representing the output image.
    The third parameter is an integer representing the number of degrees to rotate the input image. Only 90, 180, or 270 are valid inputs.
"""

inputs_info = [
    {
        "name": "img_in",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "img_out",  # Name of the result
        "array_dimension": 2,  # Number of dimensions in the result
        "shape": (height,width),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint8",
        "iotype": "out",  # Output for the kernel
    },  # Output for the kernel
    {
        "name": "direction",  # Name of the input value
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": 90,  # Minimum value for the input vector
        "max": 91,  # Maximum value for the input vector
    }  # Input vector for the kernel

]

def behavioral(img_in, direction):
    # Reshape the input image to 2D
    img_in_2d = img_in.reshape((height, width))
    
    # Create output array
    img_out_2d = np.zeros((height, width), dtype=np.uint8)
    
    # Perform rotation based on direction
    if direction == 90:
        # 90 degrees clockwise rotation
        # (x, y) -> (y, height-1-x)
        for y in range(height):
            for x in range(width):
                img_out_2d[x][height-1-y] = img_in_2d[y][x]
    
    elif direction == 180:
        # 180 degrees rotation
        # (x, y) -> (height-1-x, width-1-y)
        for y in range(height):
            for x in range(width):
                img_out_2d[height-1-y][width-1-x] = img_in_2d[y][x]
    
    elif direction == 270:
        # 270 degrees clockwise rotation (90 degrees counter-clockwise)
        # (x, y) -> (width-1-y, x)
        for y in range(height):
            for x in range(width):
                img_out_2d[width-1-x][y] = img_in_2d[y][x]
    
    else:
        # If direction is not valid, just copy the input to output
        img_out_2d = img_in_2d.copy()
    
    # Flatten the output image back to 1D array
    return img_out_2d.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")