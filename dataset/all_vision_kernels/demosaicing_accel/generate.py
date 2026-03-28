import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
TOTAL_PIXELS = HEIGHT * WIDTH
# Input: 1 byte per pixel (XF_8UC1)
# Output: 3 bytes per pixel (XF_8UC3), packed as B, G, R
TOTAL_OUT_BYTES = TOTAL_PIXELS * 3

# Hardcoded Bayer pattern: XF_BAYER_RG
# RG pattern:
#   Row 0: R G R G ...
#   Row 1: G B G B ...
BAYER_RG = 3

function_signature = "void demosaicing_accel(ap_uint<8>* img_in, ap_uint<8>* img_out)"

description = f"""Bayer demosaicing using the Malvar-He-Cutler algorithm on a {HEIGHT}x{WIDTH}
single-channel 8-bit Bayer (RG pattern) image, producing a 3-channel 8-bit color image.

Input: {TOTAL_PIXELS} uint8 values representing the Bayer mosaic image (one byte per pixel,
row-major).

Output: {TOTAL_OUT_BYTES} uint8 values representing the demosaiced color image. Each pixel
produces 3 bytes in B, G, R order, tightly packed (pixel 0 at bytes 0-2, pixel 1 at bytes 3-5,
etc.).

The RG Bayer pattern is:
  Row 0: R  G  R  G  ...
  Row 1: G  B  G  B  ...
  Row 2: R  G  R  G  ...

For each pixel, the two missing color channels are interpolated from a 5x5 neighborhood using
four integer-arithmetic kernel functions:

1. g_kernel (Green at a Red or Blue location):
   res = -(top2 + left2 + right2 + bottom2) + 2*(top1 + left1 + right1 + bottom1) + 4*center
   res = res / 8 (integer division); if res < 0, res = 0

2. rb_kernel (Red at Blue or Blue at Red location):
   t1 = (top2 + left2 + right2 + bottom2) * 3 / 2 (integer division)
   t2 = (diag_tl + diag_tr + diag_bl + diag_br) * 2
   t3 = center * 6
   res = (-t1 + t2 + t3) / 8; if res < 0, res = 0

3. rgr_bgb_kernel (Red at Green in Red row, or Blue at Green in Blue row):
   t1 = (top2 + bottom2) >> 1 (arithmetic right shift)
   t2 = diag_tl + diag_tr + left2 + right2 + diag_bl + diag_br
   t3 = (left1 + right1) * 4
   t4 = center * 5
   res = (t1 - t2 + t3 + t4) / 8; if res < 0, res = 0

4. rgb_bgr_kernel (Red at Green in Blue row, or Blue at Green in Red row):
   t1 = (left2 + right2) / 2 (integer division)
   t2 = top2 + diag_tl + diag_tr + diag_bl + diag_br + bottom2
   t3 = (top1 + bottom1) * 4
   t4 = center * 5
   res = (t1 - t2 + t3 + t4) / 8; if res < 0, res = 0

Here, top2/bottom2/left2/right2 are 2 pixels away from center in the 5x5 neighborhood,
top1/bottom1/left1/right1 are 1 pixel away, and diag_XX are diagonal neighbors.

Out-of-bounds pixels are treated as zero (constant border padding with 2-pixel border).
Output channels are clamped to [0, 255].
"""

inputs_info = [
    {
        "name": "img_in",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "uint8",
        "shape": (TOTAL_PIXELS,),
        "total_elements": TOTAL_PIXELS,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "iotype": "out",
        "array_dimension": 1,
        "dtype": "uint8",
        "shape": (TOTAL_OUT_BYTES,),
        "total_elements": TOTAL_OUT_BYTES,
        "min": 0,
        "max": 255,
    },
]


def behavioral(img_in):
    img = img_in.reshape((128, 128)).astype(np.uint8)
    bgr = cv2.cvtColor(img, cv2.COLOR_BayerRG2BGR)
    # Output is interleaved B,G,R
    return bgr.flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
