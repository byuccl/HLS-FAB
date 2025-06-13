import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 16384

function_signature = "void absolute_value_int8(int8_t *in_buffer, int8_t* out_buffer)"

description = f"""
This HLS kernel implements a function to calculate absolute value of a vector of 8-bit integer numbers. 
Parameters to the function include two int8_t pointers:
    The first pointer points to the start of a {num_values}-element array of int8_t values to be used as input data. 
    The second pointer also points to the start of a {num_values}-element array of int8_t values to be modified and viewed as the output to the function."
"""

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):  
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly  
        np.random.seed(seed)  

        input_vector = np.random.randint(-128, 127, size=(num_values), dtype=np.int8)  
        output_vector = np.zeros(num_values, dtype=np.int8)  
    inputs = [input_vector, output_vector]  

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    return np.abs(input_vector)

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")