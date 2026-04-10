import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_SIZE = 3
# KERNEL_SHAPE = XF_SHAPE_CROSS, ITERATIONS = 1
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void dilation_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies a morphological dilation to a grayscale image using a
{FILTER_SIZE}x{FILTER_SIZE} cross-shaped structuring element.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the dilated output image (same dimensions as input).

The structuring element is a {FILTER_SIZE}x{FILTER_SIZE} cross shape (center row and center
column are active; corners are inactive):
    0 1 0
    1 1 1
    0 1 0

For each output pixel at (r, c), the result is the maximum value among input pixels
covered by the active structuring element positions centered at (r, c). Out-of-bounds
positions are zero-padded (treated as 0). The dilation is applied once (ITERATIONS=1).
Output values are uint8 in the range [0, 255].
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
    se = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
    out = cv2.dilate(img, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
