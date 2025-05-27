import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Matrix dimensions    
matrix_height = 32    
matrix_width = 24    
matrix_depth = 64

description = f"""
This HLS kernel implements a function scale a given three dimensional matrix of int8 values by a given value (also int8). The scaling is done by multiplying each element by the scale factor.
Parameters to the function includes an int8 pointer and two int8 3D arrays:
    The first parameter points to the start of the 3D input data array. The dimensions are {matrix_height} x {matrix_width} x {matrix_depth}. 
    The second parameter points to a single int8 value to be used as the scale factor.
    The third parameter points to the start of the 3D output data array. The dimensions are {matrix_height} x {matrix_width} x {matrix_depth}. 
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):  
    if seed == -1: 
        data = np.load('data.npz')  
        input_matrix = data["input_matrix"]
        input_scale_factor = data["input_scale_factor"]
        output_matrix = data["output_matrix"]
    else:
        # Set a seed so the example inputs and outputs work correctly  
        np.random.seed(seed)  

        input_matrix = np.random.randint(-20, 20, size=(matrix_height, matrix_width, matrix_depth), dtype=np.int8)  
        input_scale_factor = np.random.randint(-5, 5, dtype=np.int8)  
        output_matrix = np.zeros(shape=(matrix_height, matrix_width, matrix_depth), dtype=np.int8)  
    inputs = [input_matrix, input_scale_factor, output_matrix]  

    types = generate_types(inputs)    
    return {"inputs": inputs, "types": types}    
   
# A function to generate a correct solution based on sample input vectors
def behavioral(input_matrix, input_scale_factor):    
    return input_scale_factor * input_matrix
  
examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)
    
pc.write_json("kernel.json")