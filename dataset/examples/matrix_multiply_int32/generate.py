import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the matrix
matrix1_height = 16  
matrix1_width = 32 
matrix2_height = 32  
matrix2_width = 24

description = f"""
This HLS kernel implements a function to multiply two matrices (A * B). 
Parameters to the function include three int32_t 2D arrays, all intended to be interpreted as 2-dimensional arrays:
    The first parameter points to the start of the first 2D input data array. The dimensions are {matrix1_height} x {matrix1_width}. 
    The second parameter points to the start of the second 2D input data array. The dimensions are {matrix2_height} x {matrix2_width}.
    The third parameter points to the start of the 2D output data array. The dimensions are {matrix1_height} x {matrix2_width}.
"""  
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_matrix1 = data["input_matrix1"]
        input_matrix2 = data["input_matrix2"]
        output_matrix = data["output_matrix"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        
        input_matrix1 = np.random.randint(-32767, 32767, size=(matrix1_height, matrix1_width), dtype=np.int32)
        input_matrix2 = np.random.randint(-32767, 32767, size=(matrix2_height, matrix2_width), dtype=np.int32)
        output_matrix = np.zeros(shape=(matrix1_height, matrix2_width), dtype=np.int32)
    inputs = [input_matrix1, input_matrix2, output_matrix]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input matrices  
def behavioral(input_matrix1, input_matrix2):  
    return np.matmul(input_matrix1, input_matrix2)  

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")