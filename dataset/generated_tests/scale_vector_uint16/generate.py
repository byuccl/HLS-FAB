import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 8192

description = f"""
This HLS kernel implements a function to scale a vector of 16-bit unsigned integers by a scalar value. 
Parameters to the function include two uint16_t pointers and one uint16_t variable:
    The first pointer points to the start of a {num_values}-element array of uint16_t values to be used as input data.
    The second parameter is a scalar value of type uint16_t.
    The third pointer points to the start of a {num_values}-element array of uint16_t values to be used as the output data.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        input_alpha = data["input_alpha"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        
        input_vector = np.random.randint(0, 65535, size=(num_values), dtype=np.uint16)
        input_alpha = np.random.randint(0, 65535, dtype=np.uint16)
        output_vector = np.zeros(num_values, dtype=np.uint16)
    inputs = [input_vector, input_alpha, output_vector]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_vector, alpha):
    return input_vector * alpha

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")