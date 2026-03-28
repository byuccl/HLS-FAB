import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# The Vitis equalizeHist kernel takes the input image twice (two passes):
#   img_inp  -> first pass (used to build the histogram / CDF)
#   img_inp1 -> second pass (pixels to remap via the equalization LUT)
# Both buffers contain the same image data in practice.

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void equalizeHist_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_inp1, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies histogram equalization to a grayscale image.

Parameters:
    img_inp:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
              the input image ({HEIGHT}x{WIDTH}, row-major). Used to compute the histogram.
    img_inp1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values containing
              the same input image (second pass for pixel remapping).
    img_out:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
              the equalized output image (same dimensions as input).

The kernel performs histogram equalization in two passes using the same input image:

1. First pass (img_inp): compute a 256-bin histogram H[v] = count of pixels with value v.
2. Compute the cumulative distribution function: CDF[v] = sum(H[0..v]).
3. Build a remapping LUT: LUT[v] = (CDF[v] - CDF_min) * 255 // (H*W - CDF_min)
   where CDF_min is the smallest non-zero CDF value and H*W = {HEIGHT * WIDTH}.
   Integer division (floor) is used. Values below CDF_min map to 0.
4. Second pass (img_inp1): output[i] = LUT[img_inp1[i]].

Output values are uint8 in [0, 255].
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
        "name": "img_inp1",
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

def behavioral(img_inp, img_inp1):
    img = img_inp.reshape((128, 128))
    out = cv2.equalizeHist(img)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
