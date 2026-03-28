import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 2160
WIDTH = 3840

# Scalar parameter: flip direction (0=vertical, 1=horizontal, -1=both)
DIRECTION = 1  # Horizontal flip (HOR=1 in config)

function_signature = (
    "void flip_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel flips (mirrors) a grayscale image horizontally or vertically.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The flip direction is {DIRECTION}:
- direction = 0: flip vertically (upside-down)
- direction = 1: flip horizontally (left-right mirror)
- direction = -1: flip both horizontally and vertically (180-degree rotation)

For this implementation, DIRECTION = {DIRECTION} (horizontal flip).

For horizontal flip:
- Output pixel at (i, j) = Input pixel at (i, WIDTH - 1 - j)

For vertical flip:
- Output pixel at (i, j) = Input pixel at (HEIGHT - 1 - i, j)

For both:
- Output pixel at (i, j) = Input pixel at (HEIGHT - 1 - i, WIDTH - 1 - j)
"""

inputs_info = [
    {
        "name": "img_in",
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

def behavioral(img_in):
    img = img_in.reshape((2160, 3840))
    out = cv2.flip(img, 1)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
