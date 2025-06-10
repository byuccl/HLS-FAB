import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128

description = f"""
This HLS kernel implements a function to perform bad pixel correction on an image using a 5x5 cross-diagonal mask. 
Parameters to the function include two uint8_t pointers:
    The first pointer points to the start of a {height * width}-element array of uint8_t values representing the input image.
    The second pointer points to the start of a {height * width}-element array of uint8_t values representing the output image.
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

        input_image = np.random.randint(0, 256, size=(height * width), dtype=np.uint8)
        output_image = np.zeros(height * width, dtype=np.uint8)
    inputs = [input_image, output_image]

    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

def behavioral(input_image):
    input_image_2d = input_image.reshape((height, width))
    output_image_2d = np.copy(input_image_2d)

    dx = [-2, -1, 0, 1, 2]
    dy = [-2, -1, 0, 1, 2]

    for y in range(height):
        for x in range(width):
            min_val = input_image_2d[y, x]
            max_val = input_image_2d[y, x]

            for i in range(5):
                for j in range(5):
                    if (i == 0 or i == 2 or i == 4) and (j == 0 or j == 2 or j == 4):
                        nx = x + dx[j]
                        ny = y + dy[i]
                        if 0 <= nx < width and 0 <= ny < height:
                            neighbor_val = input_image_2d[ny, nx]
                            min_val = min(min_val, neighbor_val)
                            max_val = max(max_val, neighbor_val)

            pixel_val = input_image_2d[y, x]
            if pixel_val < min_val:
                output_image_2d[y, x] = min_val
            elif pixel_val > max_val:
                output_image_2d[y, x] = max_val
            else:
                output_image_2d[y, x] = pixel_val

    return output_image_2d.flatten()

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")