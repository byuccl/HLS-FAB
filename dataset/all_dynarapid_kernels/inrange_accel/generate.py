import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameters
LOWER_THRESH = 50
UPPER_THRESH = 200

function_signature = (
    "void inrange_accel("
    "int32_t* img_in, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel performs a range check on each pixel of a grayscale image.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output binary mask image (same dimensions as input).

The kernel checks if each input pixel value falls within the range [{LOWER_THRESH}, {UPPER_THRESH}]:
- If lower_thresh <= pixel <= upper_thresh: output pixel = 255
- Otherwise: output pixel = 0

The thresholds are {LOWER_THRESH} (lower) and {UPPER_THRESH} (upper), fixed as compile-time constants.

The comparison is inclusive on both ends: pixels exactly equal to either threshold are
considered to be in range.
"""

inputs_info = [
    {
        "name": "img_in",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((128, 128))
    out = cv2.inRange(img, 50, 200)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
