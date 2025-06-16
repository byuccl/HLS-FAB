import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 524288

function_signature = "void dot_product_float32(float *in_vector1, float *in_vector2, float *result)"

description = f"""
This HLS kernel implements a function to calculate the dot product of two input vectors containing 64 values of the 32-bit floating point values."
Parameters to the function include three float32 pointers:
    The first and second pointers point to arrays containing {num_values} float values. These are the imput vectors.
    The third pointer points to a single float value to be used as the output.
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
        output_value = data["output_value"]
    else:
    # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        
        input_vector1 = np.random.rand(num_values).astype(np.float32)
        input_vector2 = np.random.rand(num_values).astype(np.float32)
        output_value = np.float32(0.0) #If a scalar output declare it like this according to the types in the kernel
    inputs = [input_vector1, input_vector2, output_value]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector1, input_vector2):
    return np.dot(input_vector1, input_vector2)

examples = generate_examples(generate_inputs)


# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")