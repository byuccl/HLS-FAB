# Example generate.py for: boxfilter
#
# Vitis Vision kernel: /home/plwalker/Vitis_Libraries/vision/L1/examples/boxfilter/
#
# Source files read to produce this:
#   - xf_box_filter_accel.cpp        -> top function name: boxfilter_accel
#   - config/xf_config_params.h      -> HEIGHT=128, WIDTH=128, IN_TYPE=XF_8UC1,
#                                       INPUT_PTR_WIDTH=8, OUTPUT_PTR_WIDTH=8,
#                                       FILTER_WIDTH=3
#   - description.json               -> topfunction: boxfilter_accel
#
# Key decisions:
#   - XF_8UC1 -> numpy dtype: uint8 (1 channel, 8 bits per pixel)
#   - img_inp / img_out are both 128x128 uint8 images
#   - "rows" and "cols" are runtime scalars -> NOT in inputs_info
#   - FILTER_WIDTH=3 is a compile-time constant -> hardcoded in behavioral()
#   - behavioral() uses scipy.ndimage.uniform_filter (box filter = uniform filter)

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3  # FILTER_SIZE_3=1 -> FILTER_WIDTH 3

function_signature = (
    "void boxfilter_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies a box filter (mean filter) to a grayscale image.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image (row-major, one byte per pixel).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for the
             filtered output image (same dimensions as input).

The kernel wraps each pixel value in an xf::cv::Mat, applies a {FILTER_WIDTH}x{FILTER_WIDTH}
box filter with XF_BORDER_CONSTANT border handling, and writes the result to the output image.
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
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.float32)
    # Box filter = mean of each FILTER_WIDTH x FILTER_WIDTH neighbourhood
    # Pad with zeros to match XF_BORDER_CONSTANT
    pad = FILTER_WIDTH // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            out[i, j] = padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH].mean()
    return out.astype(np.uint8).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
