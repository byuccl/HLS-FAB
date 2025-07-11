import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array and filter order
num_values = 1024
filter_order = 4

function_signature = "void iir_filter_float32(float *input_vector, float *a_coefficients, float *b_coefficients, float *output_vector)"

description = f"""
This HLS kernel implements an IIR filter function that applies a filter to an input vector of type float32. 
Parameters to the function include four float32 pointers:
    The first pointer points to the start of a {num_values}-element array of float32 values to be used as input data.
    The second pointer points to the start of a {filter_order + 1}-element array of float32 values representing the 'a' coefficients of the filter.
    The third pointer points to the start of a {filter_order + 1}-element array of float32 values representing the 'b' coefficients of the filter.
    The fourth pointer points to the start of a {num_values}-element array of float32 values to be used as the output data.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        a_coefficients = data["a_coefficients"]
        b_coefficients = data["b_coefficients"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        
        input_vector = np.random.rand(num_values).astype(np.float32) * 2 - 1
        a_coefficients = np.random.rand(filter_order + 1).astype(np.float32) * 2 - 1
        b_coefficients = np.random.rand(filter_order + 1).astype(np.float32) * 2 - 1
        output_vector = np.zeros(num_values, dtype=np.float32)
        
    inputs = [input_vector, a_coefficients, b_coefficients, output_vector]
    types = generate_types(inputs)  

    return {"inputs": inputs, "types": types}  

def behavioral(input_vector, a_coefficients, b_coefficients):
    output_vector = np.zeros_like(input_vector)
    for n in range(num_values):
        output_vector[n] = b_coefficients[0] * input_vector[n]
        for i in range(1, filter_order + 1):
            if n - i >= 0:
                output_vector[n] += b_coefficients[i] * input_vector[n - i] - a_coefficients[i] * output_vector[n - i]
    return output_vector

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")