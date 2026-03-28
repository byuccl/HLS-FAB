import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
OUT_HEIGHT = (HEIGHT + 1) >> 1   # = 64
OUT_WIDTH  = (WIDTH  + 1) >> 1   # = 64
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# Algorithm (from accel source / OpenCV pyrDown):
#   1. Apply 5x5 Gaussian blur with replicate border on all 4 sides.
#   2. Downsample by 2: take pixels at even rows and even columns
# Division: integer sum rounded via (sum+128)>>8.
# Border: replicate (edge) padding on all sides.
# Coefficients: standard separable Gaussian [1, 4, 6, 4, 1] / 16 per axis = /256 total.

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void pyr_down_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel downsamples a grayscale image by a factor of 2 using Gaussian pyramid
reduction (pyrDown).

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {OUT_HEIGHT * OUT_WIDTH} uint8 pixel values for
             the downsampled output image ({OUT_HEIGHT}x{OUT_WIDTH}, row-major order).

The kernel applies the following steps:

1. Convolve the input with a 5x5 Gaussian kernel using replicate (edge) border padding
   on all 4 sides: the 2 rows/columns outside each edge are filled by replicating the
   nearest edge row/column.

       [1  4  6  4  1]
       [4 16 24 16  4]   / 256
       [6 24 36 24  6]
       [4 16 24 16  4]
       [1  4  6  4  1]
   The convolution sum is rounded to nearest integer via (sum + 128) >> 8 and cast to uint8.

2. Subsample the blurred image by taking every other row and column (rows 0,2,4,...
   and columns 0,2,4,...). The output is ({OUT_HEIGHT}x{OUT_WIDTH}).
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
        "shape": (OUT_HEIGHT, OUT_WIDTH),
        "total_elements": OUT_HEIGHT * OUT_WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((128, 128))
    out = cv2.pyrDown(img, borderType=cv2.BORDER_REPLICATE)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
