import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 131072

description = f"""
This HLS kernel implements a function to calculate the standard deviation of an input vector containing 128 values of 32-bit floating-point numbers.
Parameters to the function include two float32 pointers:
    The first pointer points to an array containing {num_values} float values. This is the input vector.
    The second pointer points to a single float value to be used as the output for the standard deviation.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

# A function to return the inputs and types to the function.
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file.
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        output_std_dev = data["output_std_dev"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_vector = np.random.rand(num_values).astype(np.float32) * 100
        output_std_dev = np.float32(0.0)
    inputs = [input_vector, output_std_dev]

    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    return np.std(input_vector, dtype=np.float32)

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")