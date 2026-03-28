import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

# Scalar parameter (compile-time configuration for this generate.py)
SIGMA = 1.0

function_signature = (
    "void gaussian_filter_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies a Gaussian blur filter to a grayscale image.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The Gaussian filter smooths the image by convolving it with a {FILTER_WIDTH}x{FILTER_WIDTH}
Gaussian kernel. The kernel is computed using a Gaussian distribution with sigma = {SIGMA}.

For a {FILTER_WIDTH}x{FILTER_WIDTH} kernel with sigma = {SIGMA}, the normalized Gaussian weights are
computed using the formula:
    G(x, y) = (1 / (2*pi*sigma^2)) * exp(-(x^2 + y^2) / (2*sigma^2))

The kernel is then normalized so that all weights sum to 1.0.

Border handling uses zero-padding: pixels outside the image boundary are treated as zero.

For each output pixel:
1. Extract a {FILTER_WIDTH}x{FILTER_WIDTH} window centered on the corresponding input pixel
2. Multiply each pixel in the window by its corresponding Gaussian weight
3. Sum all weighted values
4. Round the result and store as uint8 (values are clamped to [0, 255])
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((128, 128))
    out = cv2.GaussianBlur(img, (3, 3), 1.0, borderType=cv2.BORDER_CONSTANT)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
