import numpy as np
import os
import cv2
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
channels = 3

function_signature = """void auto_exposure_correction_opencv(ap_uint<8> *input_img, ap_uint<8> *output_img)"""
description = f"""
This HLS kernel implements an auto exposure correction algorithm using luminance histogram equalization. 
The input image is converted from RGB to HSV color space, the V channel is equalized, and then the image is converted back to RGB.
Parameters to the function include two ap_uint<8> pointers:
    The first pointer points to the start of a single-row array representing a {height}x{width} image with {channels} channels. This is the input image.
    The second pointer points to the start of a single-row array representing a {height}x{width} image with {channels} channels. This is the output image.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_img = data["input_img"]
        output_img = data["output_img"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_img = np.random.randint(0, 256, size=(height, width, channels), dtype=np.uint8).flatten()
        output_img = np.zeros_like(input_img, dtype=np.uint8)
    inputs = [input_img, output_img]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_img):
    input_img = input_img.reshape((height, width, channels))
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
    hsv_img[..., 2] = cv2.equalizeHist(hsv_img[..., 2])
    output_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return output_img.flatten()

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")