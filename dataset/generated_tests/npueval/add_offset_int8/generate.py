import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 4096

function_signature = "void add_offset_int8(ap_int<8> *in_vector, ap_int<8> offset, ap_int<8> *out_vector)"

description = f"""
This HLS kernel implements a function to add an offset value to each element of an input vector of 8-bit integer numbers.
Parameters to the function include two ap_int<8> pointers and one ap_int<8> variable:
    The first pointer points to the start of a {num_values}-element array of ap_int<8> values to be used as input data.
    The second parameter is an ap_int<8> value representing the offset to be added to each element.
    The third pointer points to the start of a {num_values}-element array of ap_int<8> values to be used as the output data.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        input_offset = data["input_offset"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_vector = np.random.randint(-128, 127, size=(num_values), dtype=np.int8)
        input_offset = np.random.randint(-128, 127, dtype=np.int8)
        output_vector = np.zeros(num_values, dtype=np.int8)
    inputs = [input_vector, input_offset, output_vector]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_vector, offset):
    return input_vector + offset

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")