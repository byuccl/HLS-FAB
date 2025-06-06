import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 2160
width = 3840
channels = 3

description = f"""
This HLS kernel implements an auto white balance algorithm based on the gray-world assumption. It scales pixel values of an RGB image using a saturation threshold to determine which pixels to consider for white balancing. The algorithm stretches each channel independently to a specified range, ignoring the top and bottom 4% of pixel values for robustness.
Parameters to the function include two uint8_t pointers and four float values:
    The first parameter is a pointer to the start of the input image array, which is a single-row array representing a {height}x{width} image with {channels} channels.
    The second parameter is a pointer to the start of the output image array, which is also a single-row array representing a {height}x{width} image with {channels} channels.
    The third parameter is a float value representing the saturation threshold.
    The fourth and fifth parameters are float values representing the input minimum and maximum values for normalization.
    The sixth and seventh parameters are float values representing the output minimum and maximum values for normalization.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_img = data["input_img"]
        output_img = data["output_img"]
        input_thresh = data["input_thresh"]
        input_min = data["input_min"]
        input_max = data["input_max"]
        output_min = data["output_min"]
        output_max = data["output_max"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_img = np.random.randint(0, 256, size=(height * width * channels), dtype=np.uint8)
        output_img = np.zeros_like(input_img, dtype=np.uint8)
        input_thresh = np.float32(np.random.uniform(0.0, 1.0))
        input_min = np.float32(0.0)
        input_max = np.float32(255.0)
        output_min = np.float32(0.0)
        output_max = np.float32(255.0)
        
    inputs = [input_img, output_img, input_thresh, input_min, input_max, output_min, output_max]
    types = generate_types(inputs)  

    return {"inputs": inputs, "types": types}  

def behavioral(input_img, thresh, input_min, input_max, output_min, output_max):
    input_img = input_img.reshape((height, width, channels))
    output_img = np.zeros_like(input_img, dtype=np.uint8)
    
    sum_channels = np.zeros(channels, dtype=np.float32)
    count_channels = np.zeros(channels, dtype=np.int32)

    for i in range(height):
        for j in range(width):
            R, G, B = input_img[i, j]
            max_val = max(R, G, B)
            min_val = min(R, G, B)
            saturation = (max_val - min_val) / max_val if max_val != 0 else 0

            if saturation < thresh:
                sum_channels[0] += R
                sum_channels[1] += G
                sum_channels[2] += B
                count_channels += 1

    avg_channels = np.where(count_channels > 0, sum_channels / count_channels, 0)

    for i in range(height):
        for j in range(width):
            for c in range(channels):
                value = input_img[i, j, c]
                normalized = (value - avg_channels[c]) * (output_max - output_min) / (input_max - input_min) + output_min
                output_img[i, j, c] = np.clip(normalized, 0, 255).astype(np.uint8)

    return output_img.flatten()

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")