import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
OUT_HEIGHT = 2 * HEIGHT   # = 256
OUT_WIDTH  = 2 * WIDTH    # = 256
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# Algorithm (from accel source / Vitis pyrUp with XF_BORDER_DEFAULT = XF_BORDER_REFLECT_101):
#   1. Upsample by 2: create 2H x 2W zero array, copy input to even positions.
#   2. Apply 5x5 Gaussian kernel scaled by 4 on the 2H x 2W upsampled image.
#      Border: XF_BORDER_REFLECT_101 (same as OpenCV default / BORDER_REFLECT_101).
#      kernel = [[1,4,6,4,1],...] / 64  (= [[1,4,6,4,1],...] * 4 / 256)
#   cv2.pyrUp() uses XF_BORDER_REFLECT_101 by default, matching Vitis exactly.

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void pyr_up_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel upsamples a grayscale image by a factor of 2 using Gaussian pyramid
expansion (pyrUp).

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {OUT_HEIGHT * OUT_WIDTH} uint8 pixel values for
             the upsampled output image ({OUT_HEIGHT}x{OUT_WIDTH}, row-major order).

The kernel applies the following steps:

1. Upsample by 2: create a {OUT_HEIGHT}x{OUT_WIDTH} array of zeros. Copy each input pixel
   inp[r][c] to out[2*r][2*c]. All odd-row or odd-column positions remain zero.

2. Convolve the upsampled array with a 5x5 Gaussian kernel scaled by 4, using
   reflect-101 (BORDER_REFLECT_101) border on the 2H x 2W upsampled image.

       [1  4  6  4  1]
       [4 16 24 16  4]   / 64
       [6 24 36 24  6]
       [4 16 24 16  4]
       [1  4  6  4  1]
   The Vitis kernel uses XF_BORDER_DEFAULT = XF_BORDER_REFLECT_101, which matches
   OpenCV's default border for cv2.pyrUp().
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
    out = cv2.pyrUp(img)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
