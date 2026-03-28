import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 720
WIDTH = 1280

WINDOW_SIZE = 5
TOTAL_DISPARITY = 64
PARALLEL_UNITS = 32
NUM_DIR = 4

SMALL_PENALTY = 20
LARGE_PENALTY = 40

function_signature = (
    "void semiglobalbm_accel("
    "ap_uint<8>* img_in_l, "
    "ap_uint<8>* img_in_r, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel computes a stereo disparity map using Semi-Global Block Matching (SGBM).

Parameters:
    img_in_l: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values representing the left
              grayscale stereo image in row-major order ({HEIGHT}x{WIDTH}).
    img_in_r: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values representing the right
              grayscale stereo image in row-major order ({HEIGHT}x{WIDTH}).
    img_out:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 values for the output
              disparity map (same dimensions as inputs).

Algorithm:
1. Census Transform: For each pixel, a 5x5 census transform is computed. Each of the 24
   neighbors (excluding center) is compared against the center pixel. If the neighbor is less
   than the center, the corresponding bit is set to 1. The result is a 24-bit descriptor.
   Border pixels outside the image are treated as 0.

2. Initial Cost: For each pixel (i, j) and disparity d in [0, {TOTAL_DISPARITY}),
   the cost is the Hamming distance between the left census descriptor at (i, j) and the
   right census descriptor at (i, j-d). If j-d < 0, the right descriptor is treated as 0.

3. Semi-Global Optimization: Cost is aggregated along {NUM_DIR} directions
   (left-to-right, top-left-to-bottom-right, top-to-bottom, top-right-to-bottom-left).
   For each direction r, pixel p, and disparity d:
     Lr(p, d) = C(p, d) + min(Lr(p-r, d), Lr(p-r, d-1) + P1, Lr(p-r, d+1) + P1,
                               min_k Lr(p-r, k) + P2) - min_k Lr(p-r, k)
   where P1={SMALL_PENALTY} (penalty_small) and P2={LARGE_PENALTY} (penalty_large).
   At image boundaries for each direction, Lr(p, d) = C(p, d).

4. Cost Aggregation: S(p, d) = sum over all directions r of Lr(p, d).

5. Disparity Selection: For each pixel, the disparity with the minimum aggregated cost
   is selected. Output is uint8 (disparity value 0 to {TOTAL_DISPARITY - 1}).

Note: The HLS implementation uses 8-bit saturation in the path costs. The costs are
accumulated as 16-bit values across directions, and the minimum-cost disparity is selected.
"""

inputs_info = [
    {
        "name": "img_in_l",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in_r",
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


def behavioral(img_in_l, img_in_r):
    left = img_in_l.reshape((720, 1280)).astype(np.uint8)
    right = img_in_r.reshape((720, 1280)).astype(np.uint8)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=64,
        blockSize=5,
        P1=20,
        P2=40,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )
    disp = sgbm.compute(left, right)
    # SGBM returns fixed-point Q4: divide by 16, clip to uint8
    disp_u8 = np.clip(disp // 16, 0, 255).astype(np.uint8)
    return disp_u8.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
