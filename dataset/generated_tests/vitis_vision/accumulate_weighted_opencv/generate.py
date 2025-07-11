import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
num_values = height * width


function_signature = """void accumulate_weighted_opencv(ap_uint<8> *src1, ap_uint<8> *src2, uint16_t *dst, float alpha)"""

description = f"""
This HLS kernel implements a function to compute the weighted sum of two input images and store the result in an output image. 
Parameters to the function include two ap_uint<8> pointers, one uint16_t pointer, and a float:
    The first pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the first input image.
    The second pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the second input image.
    The third pointer points to the start of a {num_values}-element array of uint16_t values representing the output image.
    The fourth parameter is a float representing the weight alpha.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_image1 = data["input_image1"]
        input_image2 = data["input_image2"]
        output_image = data["output_image"]
        input_alpha = data["input_alpha"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image1 = np.random.randint(0, 256, size=(num_values), dtype=np.uint8)
        input_image2 = np.random.randint(0, 256, size=(num_values), dtype=np.uint8)
        output_image = np.zeros(num_values, dtype=np.uint16)
        input_alpha = np.random.default_rng().random(dtype="float32")
    inputs = [input_image1, input_image2, output_image, input_alpha]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_image1, input_image2, alpha):
    return (alpha * input_image1 + (1 - alpha) * input_image2).astype(np.uint16)

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")