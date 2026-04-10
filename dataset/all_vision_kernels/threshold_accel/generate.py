import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameters (compile-time configuration, though passed as runtime args in C++)
THRESH = 128
MAXVAL = 255

function_signature = (
    "void threshold_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies a binary threshold to a grayscale image.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The kernel applies a binary threshold operation to each pixel:
- If the input pixel value is greater than {THRESH}, the output pixel is set to {MAXVAL}
- Otherwise, the output pixel is set to 0

The threshold value is {THRESH} and the maximum value is {MAXVAL}. These are fixed
compile-time constants in this implementation.

The comparison uses strict greater-than (>), not greater-than-or-equal.
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
    _, out = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
