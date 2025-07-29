import numpy as np
import os
import cv2
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
channels = 3
num_values = height * width * channels

function_signature = "void auto_exposure_correction_opencv(ap_uint<8> *in_img, ap_uint<8> *out_img)"

description = f"""
This HLS kernel implements an auto exposure correction algorithm using luminance histogram equalization. 
The input image is converted from RGB to HSV color space, the V channel is equalized, and then the image is converted back to RGB.
Parameters to the function include two ap_uint<8> pointers:
    The first pointer points to the start of a single-row array representing a {height}x{width} image with {channels} channels. This is the input image.
    The second pointer points to the start of a single-row array representing a {height}x{width} image with {channels} channels. This is the output image.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

inputs_info = [
    {
        "name": "in_img",  # Name of the input vector
        "array_dimension": 3,  # Number of dimensions in the input vector
        "shape": (height,width,channels),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "out_img",  # Name of the result
        "array_dimension": 3,  # Number of dimensions in the result
        "shape": (height,width,channels),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint8",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

def behavioral(input_img):
    input_img = input_img.reshape((height, width, channels))
    hsv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2HSV)
    hsv_img[..., 2] = cv2.equalizeHist(hsv_img[..., 2])
    output_img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)
    return output_img.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")