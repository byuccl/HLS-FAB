import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# ARRAY = 1, SCALAR = 0
# FUNCT_NAME = absdiff (FUNCT_NUM = 0)
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# XF_8UC1 -> ap_uint<8>, dtype: uint8
# The configured function is absdiff (FUNCT_NUM=0, ARRAY=1):
#   out[i] = |in1[i] - in2[i]|  (absolute difference, two input buffers)
function_signature = (
    "void arithm_accel("
    "ap_uint<8>* img_in1, "
    "ap_uint<8>* img_in2, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel computes the per-pixel absolute difference of two grayscale images.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
             representing the first {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
             representing the second {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as inputs).

Each output pixel is computed as:
    out[i] = abs(in1[i] - in2[i])

The subtraction is performed using the widened integer type (int16) to avoid underflow,
then the absolute value is taken and the result is stored as uint8. Since the input range
is [0, 255], the result always fits in uint8 with no saturation needed.
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
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_in1, img_in2):
    img1 = img_in1.reshape((128, 128))
    img2 = img_in2.reshape((128, 128))
    out = cv2.absdiff(img1, img2)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
