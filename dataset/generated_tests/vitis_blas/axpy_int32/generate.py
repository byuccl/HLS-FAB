import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 4096

function_signature = "void axpy_int32(ap_int<32> alpha, ap_int<32> *in_vector_x, ap_int<32> *inout_vector_y)"

description = f"""
This HLS kernel implements the AXPY operation, which scales a vector X by a scalar alpha and adds it to a vector Y. 
Parameters to the function include one ap_int<32> scalar and two ap_int<32> pointers:
    The first parameter is a scalar value of type ap_int<32> representing alpha.
    The second parameter is a pointer to the start of an array of {num_values} ap_int<32> values. This is the input vector X.
    The third parameter is a pointer to the start of an array of {num_values} ap_int<32> values. This is the input/output vector Y.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_alpha = data["input_alpha"]
        input_vector_x = data["input_vector_x"]
        inout_vector_y = data["inout_vector_y"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_alpha = np.random.randint(-100, 100, dtype=np.int32)
        input_vector_x = np.random.randint(-1000, 1000, size=(num_values), dtype=np.int32)
        inout_vector_y = np.random.randint(-1000, 1000, size=(num_values), dtype=np.int32)
    inputs = [input_alpha, input_vector_x, inout_vector_y]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(alpha, input_vector_x, inout_vector_y):
    return alpha * input_vector_x + inout_vector_y

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")