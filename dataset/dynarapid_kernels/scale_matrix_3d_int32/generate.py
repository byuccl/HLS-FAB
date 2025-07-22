import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Matrix dimensions    
matrix_height = 32    
matrix_width = 24    
matrix_depth = 64

function_signature = """
void scale_matrix_3d_int32(ap_int<32> *input, ap_int<32> scale_factor, ap_int<32> *output)
// input,        [MATRIX_HEIGHT][MATRIX_WIDTH][MATRIX_DEPTH] // 3D input matrix  
// output        [MATRIX_HEIGHT][MATRIX_WIDTH][MATRIX_DEPTH] // 3D output matrix  

"""

description = f"""
This HLS kernel implements a function scale a given three dimensional matrix of int32 values by a given value (also int32). The scaling is done by multiplying each element by the scale factor.
Parameters to the function includes an int32 pointer and two int32 3D arrays:
    The first parameter points to the start of the 3D input data array. The dimensions are {matrix_height} x {matrix_width} x {matrix_depth}. 
    The second parameter points to a single int32 value to be used as the scale factor.
    The third parameter points to the start of the 3D output data array. The dimensions are {matrix_height} x {matrix_width} x {matrix_depth}. 
"""

inputs_info = [
    {
        "name": "input",  # Name of the input matrix
        "array_dimension": 3,  # Number of dimensions in the input matrix
        "shape": (matrix_height, matrix_width, matrix_depth),  # Shape of the input matrix
        "total_elements": matrix_height * matrix_width * matrix_depth,  # Total number of elements in the input matrix
        "dtype": "int32",
        "iotype": "in",  # Input matrix for the kernel
        "min": -2**15,  # Minimum value for the input matrix
        "max": 2**15 - 1,  # Maximum value for the input matrix
    },  # Input matrix for the kernel
    {
        "name": "scale_factor",  # Name of the scale factor
        "array_dimension": 0,  # Number of dimensions in the scale factor (scalar)
        "shape": (),  # Shape of the scale factor
        "total_elements": 1,  # Total number of elements in the scale factor
        "dtype": "int32",
        "iotype": "in",  # Scale factor for the kernel
        "min": -128,  # Minimum value for the scale factor
        "max": 127,  # Maximum value for the scale factor
    },  # Scale factor for the kernel
    {
        "name": "output",  # Name of the output matrix
        "array_dimension": 3,  # Number of dimensions in the output matrix
        "shape": (matrix_height, matrix_width, matrix_depth),  # Shape of the output matrix
        "total_elements": matrix_height * matrix_width * matrix_depth,  # Total number of elements in the output matrix
        "dtype": "int32",
        "iotype": "out",  # Output matrix for the kernel
    }  # Output matrix for the kernel
]
   
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