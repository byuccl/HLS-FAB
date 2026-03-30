import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
WINDOW_SIZE = 3

function_signature = (
    "void median_blur_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies a median filter to a grayscale image for noise reduction.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The median filter replaces each pixel with the median value of the {WINDOW_SIZE}x{WINDOW_SIZE}
neighborhood centered on that pixel.

For each output pixel at position (i, j):
1. Extract a {WINDOW_SIZE}x{WINDOW_SIZE} window centered at (i, j)
2. Border pixels use replicate border (edge pixels are repeated for out-of-bounds access)
3. Sort all {WINDOW_SIZE * WINDOW_SIZE} values in the window
4. Select the median value (middle value after sorting)
5. Store the median as the output pixel value (uint8)

For a {WINDOW_SIZE}x{WINDOW_SIZE} window, the median is the element at index {WINDOW_SIZE * WINDOW_SIZE // 2}
in the sorted list (0-indexed).
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
    padded = cv2.copyMakeBorder(img, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
    out = cv2.medianBlur(padded, 3)
    return out[1:-1, 1:-1].flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
