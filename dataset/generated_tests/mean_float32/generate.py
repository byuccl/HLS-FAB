import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 65536

function_signature = """void mean_float32(float *input_array, float *output_mean)"""

description = f"""
This HLS kernel implements a function to calculate the mean of an array of 32-bit floating-point numbers.
Parameters to the function include two float pointers:
    The first pointer points to the start of a {num_values}-element array of float values to be used as input data.
    The second pointer points to a single float value to be used as the output, representing the mean of the input array.
"""

# A function to return the inputs and types to the function.
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file.
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_array = data["input_array"]
        output_mean = data["output_mean"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_array = np.random.rand(num_values).astype(np.float32) * 100 - 50.0
        output_mean = np.float32(0.0)  # If a scalar output declare it like this according to the types in the kernel
    inputs = [input_array, output_mean]

    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

# A function to generate a correct solution based on sample input arrays
def behavioral(input_array):
    return np.mean(input_array)

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")