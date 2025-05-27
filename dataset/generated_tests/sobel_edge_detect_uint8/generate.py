import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
image_height = 1024
image_width = 1024

description = f"""
This HLS kernel implements the Sobel edge detection algorithm on a grayscale image of size {image_height} x {image_width} with 8-bit unsigned integer values.
Parameters to the function include two uint8_t 2D arrays:
    The first parameter points to the start of the 2D input image array. The dimensions are {image_height} x {image_width}.
    The second parameter points to the start of the 2D output image array. The dimensions are {image_height} x {image_width}.
"""

source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_image = data["input_image"]
        output_image = data["output_image"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image = np.random.randint(0, 256, size=(image_height, image_width), dtype=np.uint8)
        output_image = np.zeros((image_height, image_width), dtype=np.uint8)
    inputs = [input_image, output_image]

    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

def behavioral(input_image):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    output_image = np.zeros_like(input_image)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            region = input_image[i-1:i+2, j-1:j+2]
            sumX = np.sum(region * Gx)
            sumY = np.sum(region * Gy)
            magnitude = np.abs(sumX) + np.abs(sumY)
            output_image[i, j] = np.clip(magnitude, 0, 255)

    return output_image

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")