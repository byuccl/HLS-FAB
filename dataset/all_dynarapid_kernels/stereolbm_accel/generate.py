import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 720
WIDTH = 1280

SAD_WINDOW_SIZE = 11
NO_OF_DISPARITIES = 32
PARALLEL_UNITS = 32

# Default bm_state values from testbench
PRE_FILTER_CAP = 31
UNIQUENESS_RATIO = 15
TEXTURE_THRESHOLD = 20
MIN_DISPARITY = 0

# IN_TYPE = XF_8UC1 -> pixel width 8 -> ap_uint<8>
# OUT_TYPE = XF_16UC1 -> pixel width 16 -> ap_uint<16>
# bm_state_in is unsigned char* (not ap_uint<N>*), excluded from signature

function_signature = (
    "void stereolbm_accel("
    "int32_t* img_in_l, "
    "int32_t* img_in_r, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel computes a stereo disparity map using Local Block Matching (StereoBM).

Parameters:
    img_in_l: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values representing the left
              grayscale stereo image in row-major order ({HEIGHT}x{WIDTH}).
    img_in_r: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values representing the right
              grayscale stereo image in row-major order ({HEIGHT}x{WIDTH}).
    img_out:  Pointer to a flat array of {HEIGHT * WIDTH} uint16 values for the output
              disparity map (same dimensions as inputs).

Algorithm:
1. Sobel Pre-filtering: A 3x3 Sobel-X gradient is computed on both images. The result
   is clipped to [-preFilterCap, preFilterCap] and shifted to [0, 2*preFilterCap].
   Border pixels (first/last row and column) are set to preFilterCap.
   preFilterCap={PRE_FILTER_CAP}.

2. SAD Block Matching: For each pixel (i, j) and each disparity d in [0, {NO_OF_DISPARITIES}),
   the Sum of Absolute Differences (SAD) is computed over a {SAD_WINDOW_SIZE}x{SAD_WINDOW_SIZE}
   window between the pre-filtered left image at (i, j) and the pre-filtered right image
   at (i, j-d). The images are padded with preFilterCap for the border regions.

3. Disparity Selection: The disparity d* with the minimum SAD is selected.
   - Texture threshold: If the sum of |pixel - preFilterCap| in the left window
     is below {TEXTURE_THRESHOLD}, the pixel is marked as filtered (output=0).
   - Uniqueness ratio: If any other SAD(d) within the search range (excluding
     d*-1 and d*+1) satisfies SAD(d) <= SAD(d*) * (1 + {UNIQUENESS_RATIO}/100),
     the pixel is filtered (output=0).
   - Border pixels within half-window of the image edges or within NDISP-1 columns
     from the left are filtered.

4. Subpixel refinement: The output disparity uses 4 fractional bits:
   output = (d* * 256 + delta + 15) >> 4
   where delta = ((SAD(d*-1) - SAD(d*+1)) << 8) / k, and
   k = SAD(d*-1) + SAD(d*+1) - 2*SAD(d*) + |SAD(d*-1) - SAD(d*+1)|.
   If k==0, delta=0. For d*=0 or d*=NDISP-1, the neighbor values wrap appropriately.
"""

inputs_info = [
    {
        "name": "img_in_l",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in_r",
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


def behavioral(img_in_l, img_in_r):
    left = img_in_l.reshape((720, 1280)).astype(np.uint8)
    right = img_in_r.reshape((720, 1280)).astype(np.uint8)
    sbm = cv2.StereoBM_create(numDisparities=32, blockSize=11)
    sbm.setPreFilterCap(31)
    sbm.setUniquenessRatio(15)
    sbm.setTextureThreshold(20)
    sbm.setMinDisparity(0)
    disp = sbm.compute(left, right)
    return disp.astype(np.uint16).flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
