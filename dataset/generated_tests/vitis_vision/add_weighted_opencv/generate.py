import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
num_values = height * width

function_signature = """void add_weighted_opencv(ap_uint<8> *src1, ap_uint<8> *src2, ap_uint<8> *dst, float alpha, float beta, float gamma)"""


description = f"""
This HLS kernel implements a function to calculate a weighted sum of two input images and generates the result in the output image.
Parameters to the function include two ap_uint<8> pointers for the input images, one ap_uint<8> pointer for the output image, and three float values for the weights:
    The first pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the first input image.
    The second pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the second input image.
    The third pointer points to the start of a {num_values}-element array of ap_uint<8> values representing the output image.
    The fourth parameter is a float value representing the weight alpha.
    The fifth parameter is a float value representing the weight beta.
    The sixth parameter is a float value representing the weight gamma.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_image1 = data["input_image1"]
        input_image2 = data["input_image2"]
        output_image = data["output_image"]
        input_alpha = data["input_alpha"]
        input_beta = data["input_beta"]
        input_gamma = data["input_gamma"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image1 = np.random.randint(0, 256, size=(num_values), dtype=np.uint8)
        input_image2 = np.random.randint(0, 256, size=(num_values), dtype=np.uint8)
        output_image = np.zeros(num_values, dtype=np.uint8)
        input_alpha = np.random.uniform(0.0, 1.0)
        input_beta = np.random.uniform(0.0, 1.0)
        input_gamma = np.random.uniform(0.0, 255.0)
    inputs = [input_image1, input_image2, output_image, input_alpha, input_beta, input_gamma]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_image1, input_image2, alpha, beta, gamma):
    weighted_sum = alpha * input_image1 + beta * input_image2 + gamma
    return np.clip(weighted_sum, 0, 255).astype(np.uint8)

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")