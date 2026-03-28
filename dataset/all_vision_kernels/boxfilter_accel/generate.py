import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

function_signature = (
    "void boxfilter_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies a box filter (mean filter) to a grayscale image.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The box filter computes each output pixel as the arithmetic mean of a {FILTER_WIDTH}x{FILTER_WIDTH}
window centered on the corresponding input pixel. Border handling uses zero-padding: pixels
outside the image boundary are treated as zero.

For each output pixel at position (i, j), the kernel:
1. Extracts a {FILTER_WIDTH}x{FILTER_WIDTH} window centered at (i, j) from the zero-padded input
2. Computes the sum of all {FILTER_WIDTH * FILTER_WIDTH} values in the window (including zeros for out-of-bounds pixels)
3. Divides the sum by {FILTER_WIDTH * FILTER_WIDTH} using integer division (floor)
4. Stores the result as uint8 (values are automatically truncated/saturated to [0, 255])

Border pixels are divided by the full window size ({FILTER_WIDTH * FILTER_WIDTH}), not by the count
of in-bounds pixels only.
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
    out = cv2.boxFilter(img, -1, (3, 3), normalize=True, borderType=cv2.BORDER_CONSTANT)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
