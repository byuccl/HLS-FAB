import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
filter_size = 3

function_signature = """void bilateral_filter_opencv(ap_uint<8> *input_image, double sigma_color, double sigma_space, ap_uint<8> *output_image)"""
description = f"""
This HLS kernel implements a bilateral filter on a 128x128 image. The bilateral filter smoothens the image while preserving edges by considering both spatial and intensity differences.
Parameters to the function include two ap_uint<8> pointers and two double values:
    The first pointer points to the start of a {height * width}-element array of ap_uint<8> values representing the input image.
    The second parameter is a double value representing the sigma for color.
    The third parameter is a double value representing the sigma for space.
    The fourth pointer points to the start of a {height * width}-element array of ap_uint<8> values representing the output image.
"""

def generate_inputs(seed=0):
    if seed == -1:
        data = np.load('data.npz')
        input_image = data["input_image"]
        sigma_color = data["sigma_color"]
        sigma_space = data["sigma_space"]
        output_image = data["output_image"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image = np.random.randint(0, 256, size=(height * width), dtype=np.uint8)
        sigma_color = np.random.uniform(10.0, 50.0)
        sigma_space = np.random.uniform(10.0, 50.0)
        output_image = np.zeros(height * width, dtype=np.uint8)
    inputs = [input_image, sigma_color, sigma_space, output_image]

    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

def behavioral(input_image, sigma_color, sigma_space):
    input_image_2d = input_image.reshape((height, width))
    output_image_2d = np.zeros_like(input_image_2d)
    half_filter = filter_size // 2
    color_coeff = -0.5 / (sigma_color * sigma_color)
    space_coeff = -0.5 / (sigma_space * sigma_space)

    for y in range(height):
        for x in range(width):
            sum_weights = 0.0
            sum_intensity = 0.0
            center_pixel = input_image_2d[y, x]

            for fy in range(-half_filter, half_filter + 1):
                for fx in range(-half_filter, half_filter + 1):
                    ny, nx = y + fy, x + fx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_pixel = input_image_2d[ny, nx]
                        color_diff = center_pixel - neighbor_pixel
                        space_diff = fy * fy + fx * fx

                        weight = np.exp(space_diff * space_coeff) * np.exp(color_diff * color_diff * color_coeff)
                        sum_intensity += weight * neighbor_pixel
                        sum_weights += weight

            output_image_2d[y, x] = np.clip(sum_intensity / sum_weights, 0, 255)

    return output_image_2d.flatten()

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")