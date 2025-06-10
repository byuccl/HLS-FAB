import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples
import cv2

# Image dimensions
rows = 128
cols = 128

description = f"""
This HLS kernel implements a Sobel filter to compute the gradients of an input image in both x and y directions. 
The input image is 128x128 pixels, and each pixel is represented by an 8-bit unsigned integer.
Parameters to the function include three uint8_t pointers:
    The first pointer points to the start of a {rows * cols}-element array representing the input image.
    The second and third pointers point to the start of {rows * cols}-element arrays representing the output gradient images in the x and y directions, respectively.
The border pixels are set to a constant 0.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_image = data["input_image"]
        output_image_x = data["output_image_x"]
        output_image_y = data["output_image_y"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image = np.random.randint(0, 256, size=(rows * cols), dtype=np.uint8)
        output_image_x = np.zeros(rows * cols, dtype=np.uint8)
        output_image_y = np.zeros(rows * cols, dtype=np.uint8)
    inputs = [input_image, output_image_x, output_image_y]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_image):
    input_image_2d = input_image.reshape((rows, cols))
    grad_x = cv2.Sobel(input_image_2d, cv2.CV_16S, 1, 0, ksize=3)
    grad_y = cv2.Sobel(input_image_2d, cv2.CV_16S, 0, 1, ksize=3)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    return abs_grad_x.flatten(), abs_grad_y.flatten()

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")