import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
image_height = 128
image_width = 128

description = f"""
This HLS kernel implements a function to calculate a weighted sum of two input 128x128 images and generates the result in a destination image.
Parameters to the function include two uint8_t 2D arrays and three float variables:
    The first parameter points to the start of the first 2D input image array. The dimensions are {image_height} x {image_width}.
    The second parameter points to the start of the second 2D input image array. The dimensions are {image_height} x {image_width}.
    The third parameter is a float value representing the weight alpha for the first image.
    The fourth parameter is a float value representing the weight beta for the second image.
    The fifth parameter is a float value representing the constant gamma to be added.
    The sixth parameter points to the start of the 2D output image array. The dimensions are {image_height} x {image_width}.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1:
        data = np.load('data.npz')
        input_image1 = data["input_image1"]
        input_image2 = data["input_image2"]
        input_alpha = data["input_alpha"]
        input_beta = data["input_beta"]
        input_gamma = data["input_gamma"]
        output_image = data["output_image"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image1 = np.random.randint(0, 256, size=(image_height, image_width), dtype=np.uint8)
        input_image2 = np.random.randint(0, 256, size=(image_height, image_width), dtype=np.uint8)
        input_alpha = np.random.uniform(0.0, 1.0)
        input_beta = np.random.uniform(0.0, 1.0)
        input_gamma = np.random.uniform(-50.0, 50.0)
        output_image = np.zeros((image_height, image_width), dtype=np.uint8)
    
    inputs = [input_image1, input_image2, input_alpha, input_beta, input_gamma, output_image]
    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

def behavioral(input_image1, input_image2, alpha, beta, gamma):
    weighted_sum = input_image1.astype(np.float32) * alpha + input_image2.astype(np.float32) * beta + gamma
    weighted_sum = np.clip(weighted_sum, 0, 255)
    return weighted_sum.astype(np.uint8)

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")