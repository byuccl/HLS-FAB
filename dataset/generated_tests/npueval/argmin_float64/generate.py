import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 4096

function_signature = "void argmin_float64(double *in_vector, double *out_index)"

description = f"""
This HLS kernel implements a function to find the index of the minimum value in an input vector of float64 type.
Parameters to the function include two float64 pointers:
    The first parameter is a pointer to the start of an array of {num_values} float64 values. This is the input vector.
    The second parameter is a pointer to a single float64 value. This is the output index of the minimum value in the input vector.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        output_index = data["output_index"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        input_vector = np.random.rand(num_values).astype(np.float64) * 100
        output_index = np.float64(0.0)
        
    inputs = [input_vector, output_index]
    types = generate_types(inputs)  

    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    index = np.argmin(input_vector)
    return np.float64(index)

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")