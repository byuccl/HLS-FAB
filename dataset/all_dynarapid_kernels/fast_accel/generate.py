import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# NMS = 1 (non-maximum suppression enabled)
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# threshold is a runtime unsigned char parameter; hardcoded here.
THRESHOLD = 20

# Algorithm:
#   FAST-9 corner detector (minimum 9 consecutive pixels on Bresenham circle r=3)
#   with non-maximum suppression (NMS=1).
#   Stage 1: For each pixel, compute flag_d[j] = center - circle[j] for 16 Bresenham
#     circle pixels. Classify as corner if 9+ consecutive flag_val match (1=dark arc,
#     2=bright arc). Score = xFCoreScore = max(a0, -b0) - 1, the maximum achievable
#     threshold - 1, where a0/b0 use min/max sliding window trees over the 9-element arcs.
#   Stage 2 (NMS): Output 255 if score > 0 AND score is STRICTLY GREATER than all 8
#     neighbors in the score map. Otherwise 0.
#   Border pixels (within 3 of edge) always have score=0 → output 0.

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void fast_accel("
    "int32_t* img_in, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel detects corners in a grayscale image using the FAST (Features from
Accelerated Segment Test) algorithm with non-maximum suppression (NMS).

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the corner response map. Corner pixels are marked as 255; all others are 0.

The kernel uses FAST-9 corner detection (threshold={THRESHOLD}, NMS=1):

Stage 1 — score map:
1. For each pixel p at (r, c), compute flag_d[j] = p - circle[j] for 16 Bresenham
   circle pixels at radius 3. Classify each as 1 (flag_d > {THRESHOLD}, dark arc),
   2 (flag_d < -{THRESHOLD}, bright arc), or 0.
2. If 9+ consecutive classified pixels share the same non-zero class (checked with
   circular wraparound), the pixel is a corner candidate.
3. Compute the corner score using the xFCoreScore method: build sliding min/max trees
   over the 25-element extended flag_d array (elements 16..24 = copies of 0..8), then
   score = max(a0, -b0) - 1, where a0 is the maximum over all 9-element arcs of
   (arc minimum) and b0 is the minimum over all arcs of (arc maximum).
4. Border pixels (within 3 pixels of the image edge) always get score 0.

Stage 2 — NMS:
5. Output 255 at a pixel if its score > 0 AND is STRICTLY GREATER than all 8 neighbours
   in a 3x3 window of the score map. Otherwise output 0.
"""

inputs_info = [
    {
        "name": "img_in",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
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

def behavioral(img_in):
    img = img_in.reshape((128, 128))
    detector = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)
    kps = detector.detect(img)
    out = np.zeros((128, 128), dtype=np.uint8)
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= y < 128 and 0 <= x < 128:
            out[y, x] = 255
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
