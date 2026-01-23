import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the matrix and vector
matrix_height = 64
matrix_width = 128

function_signature = "void matrix_vector_multiply_int32(ap_int<32> *input_matrix, ap_int<32> *input_vector, ap_int<32> *output_vector)"

description = f"""
This HLS kernel implements a function to multiply a matrix by a vector, resulting in a vector. 
Parameters to the function include two int32_t arrays and one int32_t vector:
    The first parameter points to the start of the 2D input matrix. The dimensions are {matrix_height} x {matrix_width}.
    The second parameter points to the start of the 1D input vector. The dimension is {matrix_width}.
    The third parameter points to the start of the 1D output vector. The dimension is {matrix_height}.
"""


def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_matrix = data["input_matrix"]
        input_vector = data["input_vector"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        
        input_matrix = np.random.randint(-32768, 32767, size=(matrix_height, matrix_width), dtype=np.int32)
        input_vector = np.random.randint(-32768, 32767, size=(matrix_width), dtype=np.int32)
        output_vector = np.zeros(shape=(matrix_height), dtype=np.int32)
    inputs = [input_matrix, input_vector, output_vector]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_matrix, input_vector):
    return np.dot(input_matrix, input_vector)

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")