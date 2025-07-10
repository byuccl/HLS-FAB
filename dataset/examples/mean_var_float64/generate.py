import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 16384

function_signature = "void mean_var_float64(double *in_buffer, double* out_mean, double* out_var)"

description = f"""
This HLS kernel implements a function to calculate both the mean and the variance of an input vector with values of float64 type.
Parameters for this function include a float64 pointer:
    The first parameter is a float64 pointer that points to a {num_values}-value array of float64 values. 
    The second parameter is a pointer to a float64 value that will be the mean.
    The third parameter is a pointer to a float64 value that will be the variance. 
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "float64",
        "iotype": "in",  # Input buffer for the kernel
        "min": -50.0,  # Minimum value for the input buffer
        "max": 50.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_mean",  # Name of the output mean
        "array_dimension": 0,  # No dimensions, it's a scalar
        "shape": (),  # Shape of the output mean
        "total_elements": 1,  # Total number of elements in the output mean
        "dtype": "float64",
        "iotype": "out",  # Output mean for the kernel
    },  # Output mean for the kernel
    {
        "name": "out_var",  # Name of the output variance
        "array_dimension": 0,  # No dimensions, it's a scalar
        "shape": (),  # Shape of the output variance
        "total_elements": 1,  # Total number of elements in the output variance
        "dtype": "float64",
        "iotype": "out",  # Output variance for the kernel
    }  # Output variance for the kernel
]

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
# def generate_inputs(seed=0):  
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_vector = data["input_vector"]
#         output_variance = data["output_variance"]
#         output_mean = data["output_mean"]
#     else:
#         # Set a seed so the example inputs and outputs work correctly  
#         np.random.seed(seed)  

#         input_vector = np.random.rand(num_values).astype(np.float64) * 100

#         output_variance = np.float64(0.0)
#         output_mean = np.float64(0.0)
#     inputs = [input_vector, output_mean, output_variance]  

#     types = generate_types(inputs)  
#     return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector):
    mean = np.mean(input_vector, dtype=np.float64)
    var = np.var(input_vector, dtype=np.float64)
    return mean, var

examples = generate_examples(inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")