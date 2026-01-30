# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Matrix dimensions    
matrix_height = 32    
matrix_width = 24    
matrix_depth = 64

function_signature = """
void scale_matrix_3d_int8(ap_int<8> *input, ap_int<8> scale_factor, ap_int<8> *output)
// input,        [MATRIX_HEIGHT][MATRIX_WIDTH][MATRIX_DEPTH] // 3D input matrix  
// output        [MATRIX_HEIGHT][MATRIX_WIDTH][MATRIX_DEPTH] // 3D output matrix  
"""

short_description = "This HLS Kernel implements a scale matrix function with a given scale factor on 3D arrays of 8 bit integers."

description = f"""
This HLS kernel implements a function scale a given three dimensional matrix of int8 values by a given value (also int8). The scaling is done by multiplying each element by the scale factor.
Parameters to the function includes an int8 pointer and two int8 3D arrays:
    The first parameter points to the start of the 3D input data array. The dimensions are {matrix_height} x {matrix_width} x {matrix_depth}. 
    The second parameter points to a single int8 value to be used as the scale factor.
    The third parameter points to the start of the 3D output data array. The dimensions are {matrix_height} x {matrix_width} x {matrix_depth}. 
"""

inputs_info = [
    {
        "name": "input",  # Name of the input matrix
        "array_dimension": 3,  # Number of dimensions in the input matrix
        "shape": (matrix_height, matrix_width, matrix_depth),  # Shape of the input matrix
        "total_elements": matrix_height * matrix_width * matrix_depth,  # Total number of elements in the input matrix
        "dtype": "int8",
        "iotype": "in",  # Input matrix for the kernel
        "min": -20,  # Minimum value for the input matrix
        "max": 20,  # Maximum value for the input matrix
    },  # Input matrix for the kernel
    {
        "name": "scale_factor",  # Name of the scale factor
        "array_dimension": 0,  # Number of dimensions in the scale factor (scalar)
        "shape": (),  # Shape of the scale factor
        "total_elements": 1,  # Total number of elements in the scale factor
        "dtype": "int8",
        "iotype": "in",  # Scale factor for the kernel
        "min": -5,  # Minimum value for the scale factor
        "max": 5,  # Maximum value for the scale factor
    },  # Scale factor for the kernel
    {
        "name": "output",  # Name of the output matrix
        "array_dimension": 3,  # Number of dimensions in the output matrix
        "shape": (matrix_height, matrix_width, matrix_depth),  # Shape of the output matrix
        "total_elements": matrix_height * matrix_width * matrix_depth,  # Total number of elements in the output matrix
        "dtype": "int8",
        "iotype": "out",  # Output matrix for the kernel
    }  # Output matrix for the kernel
]

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
# def generate_inputs(seed=0):  
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_matrix = data["input_matrix"]
#         input_scale_factor = data["input_scale_factor"]
#         output_matrix = data["output_matrix"]
#     else:
#         # Set a seed so the example inputs and outputs work correctly  
#         np.random.seed(seed)  

#         input_matrix = np.random.randint(-20, 20, size=(matrix_height, matrix_width, matrix_depth), dtype=np.int8).flatten()
#         input_scale_factor = np.random.randint(-5, 5, dtype=np.int8)  
#         output_matrix = np.zeros(shape=(matrix_height, matrix_width, matrix_depth), dtype=np.int8).flatten() 
#     inputs = [input_matrix, input_scale_factor, output_matrix]  

#     types = generate_types(inputs)    
#     return {"inputs": inputs, "types": types}    
   
# A function to generate a correct solution based on sample input vectors
def behavioral(input_matrix, input_scale_factor): 
    input_matrix = input_matrix.reshape((matrix_height, matrix_width, matrix_depth))   
    return (input_scale_factor * input_matrix).flatten()

examples = generate_examples(inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)
    
pc.write_json("kernel.json")