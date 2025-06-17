import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 16384

function_signature = "void mean_var_float64(double *in_buffer, double* out_mean, double* out_var)"

description = f"""
This HLS kernel implements a function to calculate both the mean and the variance of an input vector with values of float64 type.
Parameters for this function include a float64 pointer:
    The first parameter is a float64 pointer that points to a {num_values}-value array of float64 values. 
    The second parameter is a pointer to a float64 value that will be the mean.
    The third parameter is a pointer to a float64 value that will be the variance. 
"""

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):  
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        output_variance = data["output_variance"]
        output_mean = data["output_mean"]
    else:
        # Set a seed so the example inputs and outputs work correctly  
        np.random.seed(seed)  

        input_vector = np.random.rand(num_values).astype(np.float64) * 100

        output_variance = np.float64(0.0)
        output_mean = np.float64(0.0)
    inputs = [input_vector, output_mean, output_variance]  

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    mean = np.mean(input_vector, dtype=np.float64)
    var = np.var(input_vector, dtype=np.float64)
    return mean, var

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")