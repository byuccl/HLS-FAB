import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 8192

function_signature = "void scale_vector_uint16(ap_uint<16> *in_vector, ap_uint<16> alpha, ap_uint<16> *out_vector)"

description = f"""
This HLS kernel implements a function to scale a vector of 16-bit unsigned integers by a scalar value. 
Parameters to the function include two ap_uint<16> pointers and one ap_uint<16> variable:
    The first pointer points to the start of a {num_values}-element array of ap_uint<16> values to be used as input data.
    The second parameter is a scalar value of type ap_uint<16>.
    The third pointer points to the start of a {num_values}-element array of ap_uint<16> values to be used as the output data.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

inputs_info = [
    {
        "name": "in_vector",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (num_values,),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": -0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "alpha",  # Name of the input vector
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "int16",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 128,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "out_vector",  # Name of the result
        "array_dimension": 1,  # Number of dimensions in the result
        "shape": (num_values,),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint16",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

def behavioral(input_vector, alpha):
    return input_vector * alpha

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")