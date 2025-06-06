import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
filter_size = 3

description = f"""
This HLS kernel implements a bilateral filter for image smoothing while preserving edges. 
Parameters to the function include two uint8_t pointers and two double values:
    The first pointer points to the start of a {height * width}-element array of uint8_t values representing the input image.
    The second parameter is a double value representing the sigma for color.
    The third parameter is a double value representing the sigma for space.
    The fourth pointer points to the start of a {height * width}-element array of uint8_t values representing the output image.
"""

source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_image = data["input_image"]
        input_sigma_color = data["input_sigma_color"]
        input_sigma_space = data["input_sigma_space"]
        output_image = data["output_image"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_image = np.random.randint(0, 256, size=(height * width), dtype=np.uint8)
        input_sigma_color = np.random.uniform(10.0, 50.0)
        input_sigma_space = np.random.uniform(10.0, 50.0)
        output_image = np.zeros(height * width, dtype=np.uint8)
    inputs = [input_image, input_sigma_color, input_sigma_space, output_image]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_image, sigma_color, sigma_space):
    half_size = filter_size // 2
    sigma_color2 = 2 * sigma_color * sigma_color
    sigma_space2 = 2 * sigma_space * sigma_space
    output_image = np.zeros_like(input_image)

    for y in range(height):
        for x in range(width):
            sum = 0.0
            w_sum = 0.0
            center_pixel = input_image[y * width + x]

            for dy in range(-half_size, half_size + 1):
                for dx in range(-half_size, half_size + 1):
                    ny = y + dy
                    nx = x + dx

                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_pixel = input_image[ny * width + nx]
                        color_diff = center_pixel - neighbor_pixel
                        spatial_weight = np.exp(-(dx * dx + dy * dy) / sigma_space2)
                        color_weight = np.exp(-(color_diff * color_diff) / sigma_color2)
                        weight = spatial_weight * color_weight

                        sum += neighbor_pixel * weight
                        w_sum += weight

            output_image[y * width + x] = np.uint8(sum / w_sum)

    return output_image

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")