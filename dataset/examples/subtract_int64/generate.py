import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 131072

description = f"""
This HLS kernel implements a function to perfom elementwise subtraction between two vectors of 64-bit integers. Each vector contains 128 elements.
Parameters to the function include three int64 pointers:
    The first and second parameters are pointers to the start of an array of {num_values} int64 values. These are the inputs.
    The third parameter is a pointer to the start of an array of {num_values} int64 values. This is the output array.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector1 = data["input_vector1"]
        input_vector2 = data["input_vector2"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_vector1 = np.random.randint(-63, 63, size=(num_values), dtype=np.int64)
        input_vector2 = np.random.randint(-63, 63, size=(num_values), dtype=np.int64)
        output_vector = np.zeros(shape=num_values, dtype=np.int64)
    inputs = [input_vector1, input_vector2, output_vector]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector1, input_vector2):
    return np.subtract(input_vector1, input_vector2)

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")