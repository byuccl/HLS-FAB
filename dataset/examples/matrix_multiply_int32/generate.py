import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the matrix
matrix1_height = 64  
matrix1_width = 32 
matrix2_height = 32  
matrix2_width = 128

function_signature = """
void matrix_multiply_int32(ap_int<32> *matrix1, ap_int<32> *matrix2, ap_int<32> *result)
//matrix1,      [MATRIX_1_HEIGHT][MATRIX_1_WIDTH]
//matrix2,      [MATRIX_2_HEIGHT][MATRIX_2_WIDTH]
//result,      [MATRIX_1_HEIGHT][MATRIX_2_WIDTH]
"""

description = f"""
This HLS kernel implements a function to multiply two matrices (A * B). 
Parameters to the function include three ap_int<32> 2D arrays, all intended to be interpreted as 2-dimensional arrays:
    The first parameter points to the start of the first 2D input data array. The dimensions are {matrix1_height} x {matrix1_width}. 
    The second parameter points to the start of the second 2D input data array. The dimensions are {matrix2_height} x {matrix2_width}.
    The third parameter points to the start of the 2D output data array. The dimensions are {matrix1_height} x {matrix2_width}.
"""  

inputs_info = [
    {
        "name": "matrix1",  # Name of the first input matrix
        "array_dimension": 2,  # Number of dimensions in the first input matrix
        "shape": (matrix1_height, matrix1_width),  # Shape of the first input matrix
        "total_elements": matrix1_height * matrix1_width,  # Total number of elements in the first input matrix
        "dtype": "int32",
        "iotype": "in",  # Input matrix for the kernel
        "min": -32767,  # Minimum value for the first input matrix
        "max": 32767,  # Maximum value for the first input matrix
    },  # First input matrix for the kernel
    {
        "name": "matrix2",  # Name of the second input matrix
        "array_dimension": 2,  # Number of dimensions in the second input matrix
        "shape": (matrix2_height, matrix2_width),  # Shape of the second input matrix
        "total_elements": matrix2_height * matrix2_width,  # Total number of elements in the second input matrix
        "dtype": "int32",
        "iotype": "in",  # Input matrix for the kernel
        "min": -32767,  # Minimum value for the second input matrix
        "max": 32767,  # Maximum value for the second input matrix
    },  # Second input matrix for the kernel
    {
        "name": "result",  # Name of the output matrix
        "array_dimension": 2,  # Number of dimensions in the output matrix
        "shape": (matrix1_height, matrix2_width),
        "total_elements": matrix1_height * matrix2_width,  # Total number of elements in the output matrix
        "dtype": "int32",
        "iotype": "out",  # Output matrix for the kernel
    }  # Output matrix for the kernel
]


# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
# def generate_inputs(seed=0):
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_matrix1 = data["input_matrix1"]
#         input_matrix2 = data["input_matrix2"]
#         output_matrix = data["output_matrix"]
#     else:
#         # Set a seed so the example inputs and outputs work correctly
#         np.random.seed(seed)
        
#         input_matrix1 = np.random.randint(-32767, 32767, size=(matrix1_height, matrix1_width), dtype=np.int32).flatten()
#         input_matrix2 = np.random.randint(-32767, 32767, size=(matrix2_height, matrix2_width), dtype=np.int32).flatten()
#         output_matrix = np.zeros(shape=(matrix1_height, matrix2_width), dtype=np.int32).flatten()
#     inputs = [input_matrix1, input_matrix2, output_matrix]

#     types = generate_types(inputs)  
#     return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input matrices  
def behavioral(input_matrix1, input_matrix2):  
    input_matrix1 = input_matrix1.reshape((matrix1_height, matrix1_width))  
    input_matrix2 = input_matrix2.reshape((matrix2_height, matrix2_width))
    return np.matmul(input_matrix1, input_matrix2).flatten()  

examples = generate_examples(inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")