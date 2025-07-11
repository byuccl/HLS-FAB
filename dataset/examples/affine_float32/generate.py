import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

num_values = 16384

function_signature = "void affine_float32(float *in_vector, float in_scale, float in_bias, float *out_vector)"

description = f"""
This HLS kernel implements a function to calculate the affine transform of a vector of 32-bit floating point type numbers. 
Parameters to the function include 2 float32 pointers and 2 float32 variables:
    The first pointer points to the start of a {num_values}-element array of float32 values to be used as input data. 
    The second parameter is a scale factor of type float32
    The third parameter is a bias factor of type float32.
    The fourth parameter is a pointer that points to the start of a {num_values}-element array of float32 values to be used as the output data.
"""

inputs_info = [
    {
        "name": "in_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "float32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -50.0,  # Minimum value for the input buffer
        "max": 50.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "in_scale",  # Name of the scale factor
        "array_dimension": 0,  # No dimensions, it's a scalar
        "shape": (),  # Shape of the scale factor
        "total_elements": 1,  # Total number of elements in the scale factor
        "dtype": "float32",
        "iotype": "in",  # Input scalar for the kernel
        "min": -50.0,  # Minimum value for the scale factor
        "max": 50.0,  # Maximum value for the scale factor
    },  # Scale factor for the kernel
    {
        "name": "in_bias",  # Name of the bias factor
        "array_dimension": 0,  # No dimensions, it's a scalar
        "shape": (),  # Shape of the bias factor
        "total_elements": 1,  # Total number of elements in the bias factor
        "dtype": "float32",
        "iotype": "in",  # Input scalar for the kernel
        "min": -50.0,  # Minimum value for the bias factor
        "max": 50.0,  # Maximum value for the bias factor
    },  # Bias factor for the kernel
    {
        "name": "out_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "float32",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]


# def generate_inputs(seed=0):
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_vector = data["input_vector"]
#         input_scale = data["input_scale"]
#         input_bias = data["input_bias"]
#         output_vector = data["output_vector"]
#     else:
#         # Set a seed so the example inputs and outputs work correctly
#         np.random.seed(seed)
#         input_vector = np.random.rand(num_values).astype(np.float32) * 100 - 50.0
#         # Generate a random scale and bias
#         input_scale = np.float32(np.random.uniform(-50.0, 50.0))
#         input_bias = np.float32(np.random.uniform(-50.0, 50.0))
#         output_vector = np.zeros(num_values).astype(np.float32)
#     inputs = [input_vector, input_scale, input_bias, output_vector]
#     types = generate_types(inputs)  
#     return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector, scale, bias):
    output_vector = np.zeros_like(input_vector)
    for i in range(len(input_vector)):
        output_vector[i] = input_vector[i] * scale + bias
    return output_vector

examples = generate_examples(inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")