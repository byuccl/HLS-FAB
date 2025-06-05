import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 4096

description = f"""
This HLS kernel implements the AXPY operation on two input vectors of 32-bit integers. 
The operation is defined as Y = alpha * X + Y, where X and Y are vectors and alpha is a scalar.
Parameters to the function include two int32_t pointers and one int32_t scalar:
    The first pointer points to an array containing {num_values} int32 values. This is the input vector X.
    The second parameter is an int32 scalar value, alpha.
    The third pointer points to an array containing {num_values} int32 values. This is the input/output vector Y.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_x_vector = data["input_x_vector"]
        input_alpha = data["input_alpha"]
        input_y_vector = data["input_y_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_x_vector = np.random.randint(-1000, 1000, size=(num_values), dtype=np.int32)
        input_alpha = np.random.randint(-10, 10, dtype=np.int32)
        input_y_vector = np.random.randint(-1000, 1000, size=(num_values), dtype=np.int32)
    inputs = [input_x_vector, input_alpha, input_y_vector]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_x_vector, input_alpha, input_y_vector):
    return input_alpha * input_x_vector + input_y_vector

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")