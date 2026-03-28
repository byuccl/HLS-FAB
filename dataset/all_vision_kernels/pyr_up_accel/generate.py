import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
OUT_HEIGHT = 2 * HEIGHT   # = 256
OUT_WIDTH  = 2 * WIDTH    # = 256
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# Algorithm (from accel source / OpenCV pyrUp):
#   1. Upsample by 2: create 2H x 2W zero array, copy input to even positions.
#   2. Apply 5x5 Gaussian kernel scaled by 4 on the 2H x 2W upsampled image with mixed
#      border: replicate top/left/right, zero bottom. (sum+32)>>6; cast to uint8 (wrap-around).
#      kernel = [[1,4,6,4,1],...] / 64  (= [[1,4,6,4,1],...] * 4 / 256)
#   Border on the upsampled image: replicate top/left/right, zero bottom.
#   Output wraps around (no saturation) when result > 255 near replicated borders.

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

2. Convolve the upsampled array with a 5x5 Gaussian kernel scaled by 4, using mixed border
   on the 2H x 2W upsampled image:
   - Top border: replicate row 0 of the upsampled image for the 2 rows above.
   - Left and right borders: replicate the leftmost/rightmost column.
   - Bottom border: zero-pad.

       [1  4  6  4  1]
       [4 16 24 16  4]   / 64
       [6 24 36 24  6]
       [4 16 24 16  4]
       [1  4  6  4  1]
   The convolution sum is computed as (sum + 32) >> 6 and cast to uint8 with wrap-around
   (no saturation). Near replicated borders, the sum can exceed 255 and wraps modulo 256.
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
