import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# OUT_TYPE = XF_32SC1 -> ap_uint<32>, dtype: int32
# INPUT_PTR_WIDTH = 8, OUTPUT_PTR_WIDTH = 32

function_signature = (
    "void integral_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<32>* img_out)"
)

description = f"""
This HLS kernel computes the 2D integral image (summed area table) of a grayscale input.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} int32 values for the integral
             image output ({HEIGHT}x{WIDTH}), in row-major order.

The kernel computes the standard 2D prefix sum. For each output pixel at (row, col):
    out[row, col] = sum of all input pixels at positions (r, c) where r <= row and c <= col.

Implementation detail: a line buffer holds the running column-prefix sum from the previous
row. For each pixel, a running row-prefix sum is accumulated left-to-right; it is then
added to the corresponding column-prefix from the line buffer to give the 2D prefix sum,
which is also written back to the line buffer for the next row.

Output is stored as int32 (no saturation or overflow protection for arrays larger than
what int32 can hold; for 128x128 uint8 images the maximum value is 255*128*128=4,177,920
which is well within int32 range).
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
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((128, 128))
    result = cv2.integral(img, sdepth=cv2.CV_32S)
    return result[1:, 1:].astype(np.int32).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
