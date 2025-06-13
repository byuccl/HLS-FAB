import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 262144

function_signature = "void relu_activation_float32(float *in_vector, float *out_vector)"

description = f"""
This HLS kernel implements the ReLU activation function on a input vector of 64 values of type 32-bit floating-point.
Parameters to the function include two float32 pointers:
    The first pointer points to an array containing {num_values} float values. This is the input vector.
    The second pointer also points to an array containing {num_values} float values. This is the output vector.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):  
    if seed == -1: 
        data = np.load('data.npz')  
        input_vector = data["input_vector"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly  
        np.random.seed(seed)  

        input_vector = np.random.rand(num_values).astype(np.float32) * 10.0 - 5.0  
        output_vector = np.zeros(num_values, dtype=np.float32)  
    inputs = [input_vector, output_vector]  

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    return np.maximum(0, input_vector)

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")