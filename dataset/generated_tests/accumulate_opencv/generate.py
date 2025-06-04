import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
num_values = height * width

description = f"""
This HLS kernel implements a function to accumulate two images. 
Parameters to the function include two uint8_t pointers and one uint16_t pointer:
    The first pointer points to the start of a {num_values}-element array of uint8_t values representing the first input image.
    The second pointer points to the start of a {num_values}-element array of uint8_t values representing the second input image.
    The third pointer points to the start of a {num_values}-element array of uint16_t values representing the output accumulated image.
"""

source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_image1 = data["input_image1"]
        input_image2 = data["input_image2"]
        output_image = data["output_image"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image1 = np.random.randint(0, 256, size=(num_values), dtype=np.uint8)
        input_image2 = np.random.randint(0, 256, size=(num_values), dtype=np.uint8)
        output_image = np.zeros(num_values, dtype=np.uint16)
    inputs = [input_image1, input_image2, output_image]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_image1, input_image2):
    return np.add(input_image1.astype(np.uint16), input_image2.astype(np.uint16))

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")