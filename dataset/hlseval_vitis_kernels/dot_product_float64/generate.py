import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Set the number of values in the array
num_values = 1024

function_signature = "void dot_product_float64(double *in_vector1, double *in_vector2, double *result)"

description = f"""
This HLS kernel implements a function to calculate the dot product of two input vectors containing {num_values} values of the 64-bit floating point values.
Parameters to the function include three float64 pointers:
    The first and second pointers point to arrays containing {num_values} float values. These are the input vectors.
    The third pointer points to a single float value to be used as the output.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

inputs_info = [
    {
        "name": "in_vector1",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (num_values,),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "float64",
        "iotype": "in",  # Input vector for the kernel
        "min": -32000,  # Minimum value for the input vector
        "max": 32000,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "in_vector2",  # Name of the input vector
        "array_dimension": 1,  # Number of dimensions in the input vector
        "shape": (num_values,),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "float64",
        "iotype": "in",  # Input vector for the kernel
        "min": -32000,  # Minimum value for the input vector
        "max": 32000,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "result",  # Name of the result
        "array_dimension": 0,  # Number of dimensions in the result
        "shape": (),  # Shape of the result
        "total_elements": 1,  # Total number of elements in the result
        "dtype": "float64",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

# A function to generate a correct solution based on sample input vectors
def behavioral(input_vector1, input_vector2):
    return np.dot(input_vector1, input_vector2)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")