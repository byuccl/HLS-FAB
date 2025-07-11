import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the image
height = 128
width = 128

function_signature = """void reduce_avg_opencv(ap_uint<8> *input_image, int dimension, ap_uint<8> *output_vector)"""

description = f"""
This HLS kernel implements a function to reduce a 128x128 image to a vector by calculating the mean of its rows or columns.
Parameters to the function include two ap_uint<8> pointers and an integer:
    The first pointer points to the start of a {height}x{width} array of ap_uint<8> values representing the input image.
    The second parameter is an integer representing the dimension along which the reduction is performed (0 for rows, 1 for columns).
    The third pointer points to the start of an array of ap_uint<8> values representing the output vector.
"""


def generate_inputs(seed=0):
    if seed == -1:
        data = np.load('data.npz')
        input_image = data["input_image"]
        dimension = data["dimension"]
        output_vector = data["output_vector"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
        dimension = np.random.choice([0, 1])
        output_vector = np.zeros(height if dimension == 1 else width, dtype=np.uint8)

    inputs = [input_image.flatten(), dimension, output_vector]
    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

def behavioral(input_image, dimension):
    input_image = input_image.reshape((height, width))
    if dimension == 0:
        return np.mean(input_image, axis=0).astype(np.uint8)
    else:
        return np.mean(input_image, axis=1).astype(np.uint8)

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")