import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameter
ALPHA = 0.76

function_signature = (
    "void accumulate_weighted_accel("
    "ap_uint<8>* img_in1, "
    "ap_uint<8>* img_in2, "
    "ap_uint<16>* img_out)"
)

description = f"""
This HLS kernel computes a weighted sum of two grayscale images.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the first {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the second {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values for
             the output image (same dimensions as inputs).

The kernel computes the weighted sum using the formula:
    img_out[i] = alpha * img_in1[i] + (1 - alpha) * img_in2[i]

Where alpha = {ALPHA} is a fixed compile-time constant.

For each pixel:
1. Multiply img_in1[i] by alpha ({ALPHA})
2. Multiply img_in2[i] by (1 - alpha) ({1.0 - ALPHA})
3. Add the two weighted values
4. Truncate (floor) the result to the nearest integer and store as uint16

The uint16 output type accommodates the full range of possible results. The computation
uses floating-point arithmetic internally, then converts the final result to uint16 by
truncation (floor cast).
"""

inputs_info = [
    {
        "name": "img_in1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in2",
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
        "dtype": "uint16",
        "iotype": "out",
    },
]

def behavioral(img_in1, img_in2):
    a = img_in1.astype(np.float64)
    b = img_in2.astype(np.float64)
    return np.clip(0.76 * a + 0.24 * b, 0, 65535).astype(np.uint16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
