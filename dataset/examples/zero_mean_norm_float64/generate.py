import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 65536

function_signature = "void zero_mean_norm_float64(double *in_vector, double *out_vector, double *mean)"

description = f"""
This HLS kernel implements a function that does a zero mean normalization on an input vector containing {num_values} values of float64 type.
Parameters to the function include three float64 pointers:
    The first parameter is a pointer to the start of an array of {num_values} float64 values. This is the input array.
    The second parameter is a pointer to the start of an array of {num_values} float64 values. This is the output array for the normalized outputs.
    The third parameter is a pointer a single float64 value. This value is the mean, which must be calculated and is another output to the function.
"""

inputs_info = [
    {
        "name": "in_vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (num_values,),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "float64",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 100,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "out_vector",  # Name of the output vector
        "array_dimension": 1,  # Number of dimensions in the output vector
        "shape": (num_values,),  # Shape of the output vector
        "total_elements": num_values,  # Total number of elements in the output vector
        "dtype": "float64",
        "iotype": "out",  # Output vector for the kernel
    },  # Output vector for the kernel
    {
        "name": "mean",  # Name of the output mean
        "array_dimension": 0,  # No dimensions, it's a scalar
        "shape": (),  # Shape of the output mean
        "total_elements": 1,  # Total number of elements in the output mean
        "dtype": "float64",
        "iotype": "out",  # Output mean for the kernel
    }  # Output mean for the kernel
]   

# def generate_inputs(seed=0):
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_vector = data["input_vector"]
#         output_vector = data["output_vector"]
#         output_mean = data["output_mean"]
#     else:
#         # Set a seed so the example inputs and outputs work correctly
#         np.random.seed(seed)
#         input_vector = np.random.rand(num_values).astype(np.float64) * 100

#         output_vector = np.zeros(num_values).astype(np.float64)
#         output_mean = np.float64(0.0)
        
#     inputs = [input_vector, output_vector, output_mean]
#     types = generate_types(inputs)  

#     return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    mean = np.mean(input_vector)
    output_vector = (input_vector - mean)
    return output_vector, mean

examples = generate_examples(inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")