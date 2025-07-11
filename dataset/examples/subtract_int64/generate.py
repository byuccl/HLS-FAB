import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 16384

function_signature = "void subtract_int64(ap_int<64> *in_buffer1, ap_int<64> *in_buffer2, ap_int<64>* out_buffer)"

description = f"""
This HLS kernel implements a function to perfom elementwise subtraction between two vectors of 64-bit integers. Each vector contains {num_values} elements.
Parameters to the function include three ap_int<64> pointers:
    The first and second parameters are pointers to the start of an array of {num_values} ap_int<64> values. These are the inputs.
    The third parameter is a pointer to the start of an array of {num_values} ap_int<64> values. This is the output array.
"""

inputs_info = [
    {
        "name": "in_buffer1",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the first input buffer
        "shape": (num_values,),  # Shape of the first input buffer
        "total_elements": num_values,  # Total number of elements in the first input buffer
        "dtype": "int64",
        "iotype": "in",  # Input buffer for the kernel
        "min": -63,  # Minimum value for the first input buffer
        "max": 63,  # Maximum value for the first input buffer
    },  # First input buffer for the kernel
    {
        "name": "in_buffer2",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the second input buffer
        "shape": (num_values,),  # Shape of the second input buffer
        "total_elements": num_values,  # Total number of elements in the second input buffer
        "dtype": "int64",
        "iotype": "in",  # Input buffer for the kernel
        "min": -63,  # Minimum value for the second input buffer        
        "max": 63,  # Maximum value for the second input buffer
    },  # Second input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "int64",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
# def generate_inputs(seed=0):
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_vector1 = data["input_vector1"]
#         input_vector2 = data["input_vector2"]
#         output_vector = data["output_vector"]
#     else:
#         # Set a seed so the example inputs and outputs work correctly
#         np.random.seed(seed)

#         input_vector1 = np.random.randint(-63, 63, size=(num_values), dtype=np.int64)
#         input_vector2 = np.random.randint(-63, 63, size=(num_values), dtype=np.int64)
#         output_vector = np.zeros(shape=num_values, dtype=np.int64)
#     inputs = [input_vector1, input_vector2, output_vector]

#     types = generate_types(inputs)  
#     return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector1, input_vector2):
    return np.subtract(input_vector1, input_vector2)

examples = generate_examples(inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")