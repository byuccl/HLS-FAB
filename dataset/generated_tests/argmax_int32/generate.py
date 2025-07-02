import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 4096

function_signature = "void argmax_int32(ap_int<32> *in_vector, ap_int<32> *out_index)"

description = f"""
This HLS kernel implements a function to find the index of the maximum value in an input vector of 32-bit integers.
Parameters to the function include two ap_int<32> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_int<32> values to be used as input data.
    The second pointer points to a single ap_int<32> value to be used as the output index of the maximum value in the input array.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        output_index = data["output_index"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_vector = np.random.randint(-2147483648, 2147483647, size=(num_values), dtype=np.int32)
        output_index = np.int32(0)  # Initialize output index
    inputs = [input_vector, output_index]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_vector):
    return np.argmax(input_vector)

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")