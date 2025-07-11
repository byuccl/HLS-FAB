import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 16384

function_signature = "void dot_product_float32(float *in_vector1, float *in_vector2, float *result)"

description = f"""
This HLS kernel implements a function to calculate the dot product of two input vectors containing {num_values} values of the 32-bit floating point values."
Parameters to the function include three float32 pointers:
    The first and second pointers point to arrays containing {num_values} float values. These are the input vectors.
    The third pointer points to a single float value to be used as the output.
"""

inputs_info = [
    {
        "name": "in_vector1",  # Name of the first input vector
        "array_dimension": 1,  # Number of dimensions in the first input vector 
        "shape": (num_values,),  # Shape of the first input vector
        "total_elements": num_values,  # Total number of elements in the first input vector
        "dtype": "float32",
        "iotype": "in",  # Input vector for the kernel
        "min": -50.0,  # Minimum value for the first input vector
        "max": 50.0,  # Maximum value for the first input vector
    },  # First input vector for the kernel
    {
        "name": "in_vector2",  # Name of the second input vector
        "array_dimension": 1,  # Number of dimensions in the second input vector
        "shape": (num_values,),  # Shape of the second input vector
        "total_elements": num_values,  # Total number of elements in the second input vector
        "dtype": "float32",
        "iotype": "in",  # Input vector for the kernel  
        "min": -50.0,  # Minimum value for the second input vector
        "max": 50.0,  # Maximum value for the second input vector
    },  # Second input vector for the kernel
    {
        "name": "result",  # Name of the output value
        "array_dimension": 0,  # No dimensions, it's a scalar
        "shape": (),  # Shape of the output value
        "total_elements": 1,  # Total number of elements in the output value
        "dtype": "float32",
        "iotype": "out",  # Output value for the kernel
    }  # Output value for the kernel
]


# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
# def generate_inputs(seed=0):
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_vector1 = data["input_vector1"]
#         input_vector2 = data["input_vector2"]
#         output_value = data["output_value"]
#     else:
#     # Set a seed so the example inputs and outputs work correctly
#         np.random.seed(seed)
        
#         input_vector1 = np.random.rand(num_values).astype(np.float32)
#         input_vector2 = np.random.rand(num_values).astype(np.float32)
#         output_value = np.float32(0.0) #If a scalar output declare it like this according to the types in the kernel
#     inputs = [input_vector1, input_vector2, output_value]

#     types = generate_types(inputs)  
#     return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector1, input_vector2):
    return np.dot(input_vector1, input_vector2)

examples = generate_examples(inputs_info)


# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")