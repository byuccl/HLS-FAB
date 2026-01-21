import numpy as np
import os
from hlseval import PromptConstructor, generate_examples

# Image dimensions
height = 128
width = 128
num_values = height * width

function_signature = "void accumulate_vitis(ap_uint<8> *src1, ap_uint<8> *src2, ap_uint<16> *dst)"

description = f"""
This HLS kernel implements a function to accumulate two images. 
Parameters to the function include two ap_uint<8> pointers and one ap_uint<16> pointer:
    The first pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the first input image.
    The second pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the second input image.
    The third pointer points to the start of a {num_values}-element array of ap_uint<16> values representing the output accumulated image.
"""

inputs_info = [
    {
        "name": "src1",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "src2",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "result",  # Name of the result
        "array_dimension": 2,  # Number of dimensions in the result
        "shape": (height,width),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint16",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def behavioral(input_image1, input_image2):
    return np.add(input_image1.astype(np.uint16), input_image2.astype(np.uint16))

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")