import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the matrix
matrix_size = 128

function_signature = """void reduce_avg_uint8(ap_uint<8> *input_matrix, ap_uint<8> *output_vector)"""
description = f"""
This HLS kernel implements a function to reduce a matrix to a vector by calculating the mean of each column. 
Parameters to the function include a ap_uint<8> 2D array and a ap_uint<8> 1D array:
    The first parameter points to the start of the 2D input data array. The dimensions are {matrix_size} x {matrix_size}. 
    The second parameter points to the start of the 1D output data array. The length is {matrix_size}.
"""


def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_matrix = data["input_matrix"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_matrix = np.random.randint(0, 256, size=(matrix_size, matrix_size), dtype=np.uint8)
        output_vector = np.zeros(matrix_size, dtype=np.uint8)
    inputs = [input_matrix, output_vector]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_matrix):
    return np.mean(input_matrix, axis=0).astype(np.uint8)

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")