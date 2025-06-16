import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 65536

description = f"""
This HLS kernel implements a function that does a zero mean normalization on an input vector containing 64 values of float64 type.
Parameters to the function include three float64 pointers:
    The first parameter is a pointer to the start of an array of {num_values} float64 values. This is the input array.
    The second parameter is a pointer to the start of an array of {num_values} float64 values. This is the output array for the normalized outputs.
    The third parameter is a pointer a single float64 value. This value is the mean, which must be calculated and is another output to the function.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        output_vector = data["output_vector"]
        output_mean = data["output_mean"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        input_vector = np.random.rand(num_values).astype(np.float64) * 100

        output_vector = np.zeros(num_values).astype(np.float64)
        output_mean = np.float64(0.0)
        
    inputs = [input_vector, output_vector, output_mean]
    types = generate_types(inputs)  

    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    mean = np.mean(input_vector)
    output_vector = (input_vector - mean)
    return output_vector, mean

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")