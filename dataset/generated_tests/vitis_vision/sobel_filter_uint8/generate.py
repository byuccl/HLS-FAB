import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
image_height = 128
image_width = 128

function_signature = """void sobel_filter_uint8(ap_uint<8> *input_image, ap_int<16> *output_image_x, ap_int<16> *output_image_y)"""

description = f"""
This HLS kernel implements a Sobel filter to compute the gradients of a 128x128 input image in both x and y directions by convolving the kernel with the input image.
Parameters to the function include one ap_uint<8> 2D array and two ap_int<16> 2D arrays:
    The first parameter points to the start of the 2D input image array. The dimensions are {image_height} x {image_width}.
    The second parameter points to the start of the 2D output image array for the x-direction gradients. The dimensions are {image_height} x {image_width}.
    The third parameter points to the start of the 2D output image array for the y-direction gradients. The dimensions are {image_height} x {image_width}.
"""


def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_image = data["input_image"]
        output_image_x = data["output_image_x"]
        output_image_y = data["output_image_y"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image = np.random.randint(0, 256, size=(image_height, image_width), dtype=np.uint8)
        output_image_x = np.zeros((image_height, image_width), dtype=np.int16)
        output_image_y = np.zeros((image_height, image_width), dtype=np.int16)
    inputs = [input_image, output_image_x, output_image_y]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_image):
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int8)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int8)

    output_image_x = np.zeros_like(input_image, dtype=np.int16)
    output_image_y = np.zeros_like(input_image, dtype=np.int16)

    for i in range(1, image_height - 1):
        for j in range(1, image_width - 1):
            region = input_image[i-1:i+2, j-1:j+2]
            gradient_x = np.sum(region * sobel_x)
            gradient_y = np.sum(region * sobel_y)
            output_image_x[i, j] = gradient_x
            output_image_y[i, j] = gradient_y

    return output_image_x, output_image_y

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")