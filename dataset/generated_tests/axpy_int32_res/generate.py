import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 4096

description = f"""
This HLS kernel implements the AXPY operation on vectors of 32-bit integers. 
Parameters to the function include one int32 scalar and three int32 pointers:
    The first parameter is a scalar value of type int32, representing the alpha multiplier.
    The second parameter points to an array containing {num_values} int32 values. This is the input vector X.
    The third parameter points to an array containing {num_values} int32 values. This is the input vector Y.
    The fourth parameter points to an array containing {num_values} int32 values. This is the output vector Y_res, which stores the result of the operation.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_alpha = data["input_alpha"]
        input_x_vector = data["input_x_vector"]
        input_y_vector = data["input_y_vector"]
        output_y_res = data["output_y_res"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)
        
        input_alpha = np.random.randint(-100, 100, dtype=np.int32)
        input_x_vector = np.random.randint(-1000, 1000, size=(num_values), dtype=np.int32)
        input_y_vector = np.random.randint(-1000, 1000, size=(num_values), dtype=np.int32)
        output_y_res = np.zeros(num_values, dtype=np.int32)
    inputs = [input_alpha, input_x_vector, input_y_vector, output_y_res]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(alpha, x_vector, y_vector):
    return alpha * x_vector + y_vector

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")