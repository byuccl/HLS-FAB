import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameters
SCALE = 1.5
SHIFT = 10.0

function_signature = (
    "void convertScaleAbs_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel scales, shifts, and takes the absolute value of each pixel in a grayscale image.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The kernel computes each output pixel using the formula:
    out[i] = saturate_cast<uint8>(abs(scale * img_in[i] + shift))

Where:
- scale = {SCALE}
- shift = {SHIFT}
- abs() takes the absolute value
- saturate_cast clamps the result to the range [0, 255]

Processing steps for each pixel:
1. Multiply input pixel by scale: temp = {SCALE} * pixel
2. Add shift: temp = temp + {SHIFT}
3. Take absolute value: temp = abs(temp)
4. Saturate to [0, 255] and convert to uint8

The saturation ensures that:
- Values less than 0 become 0
- Values greater than 255 become 255
- Values in [0, 255] are truncated (floor) to the nearest integer
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
    img = img_in.reshape((128, 128))
    out = cv2.convertScaleAbs(img, alpha=1.5, beta=10.0)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
