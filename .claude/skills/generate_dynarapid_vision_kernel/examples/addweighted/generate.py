# Example generate.py for: addweighted (DynaRapid int32 version)
#
# Vitis Vision kernel: ${VITIS_LIBRARIES}/vision/L1/examples/addweighted/
#
# Source files read to produce this:
#   - xf_add_weighted_accel.cpp       -> top function name: add_weighted_accel
#   - config/xf_config_params.h       -> HEIGHT=128, WIDTH=128, IN_TYPE=XF_8UC1
#   - description.json                -> topfunction: add_weighted_accel
#
# DynaRapid adaptation:
#   - Original XF_8UC1 (uint8, range 0-255) → int32_t (range 0-255 in int32 container)
#   - All pointer params → int32_t*
#   - Scalar params (alpha, beta, gamma) remain as hardcoded constants
#   - behavioral() clamps output to [0, 255]
#   - Function name: add_weighted_accel_int32
#
# This kernel demonstrates:
#   - Multiple inputs (two images) with scalar parameters
#   - Scalar params NOT in inputs_info (hardcoded as module-level constants)
#   - Weighted blend with float arithmetic, result truncated to integer range
#
# C++ source analysis:
#   - Computation: alpha * in1 + beta * in2 + gamma
#   - Intermediate: float32 arithmetic
#   - Output: truncated (floor cast) to uint8 range [0, 255], saturated

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# --- Logical pixel range ---
# Original Vitis type: XF_8UC1 → logical range [0, 255]
PIXEL_MIN = 0
PIXEL_MAX = 255

# Scalar parameters (not in inputs_info — hardcoded constants)
ALPHA = 0.5
BETA  = 0.5
GAMMA = 0.0

function_signature = (
    "void add_weighted_accel_int32("
    "int32_t* img_in1, "
    "int32_t* img_in2, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel computes a weighted blend of two grayscale images.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} int32 values representing
             the first {HEIGHT}x{WIDTH} grayscale input image (row-major).
             Each value represents a pixel in the range [{PIXEL_MIN}, {PIXEL_MAX}].
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} int32 values representing
             the second {HEIGHT}x{WIDTH} grayscale input image (row-major).
             Each value represents a pixel in the range [{PIXEL_MIN}, {PIXEL_MAX}].
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} int32 values for
             the blended output image (same dimensions as inputs).
             Each output value is clamped to [{PIXEL_MIN}, {PIXEL_MAX}].

Each output pixel is computed as:
    out = clip(alpha * in1 + beta * in2 + gamma, {PIXEL_MIN}, {PIXEL_MAX})

where alpha={ALPHA}, beta={BETA}, gamma={GAMMA} are fixed compile-time constants.
The intermediate result is computed in float32, then truncated (floor cast) to integer.
Values below {PIXEL_MIN} are clamped to {PIXEL_MIN}; values above {PIXEL_MAX} are
clamped to {PIXEL_MAX}. Pixel values are logically 8-bit unsigned (0-255) stored in int32.
"""

inputs_info = [
    {
        "name": "img_in1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": PIXEL_MIN,
        "max": PIXEL_MAX,
    },
    {
        "name": "img_in2",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": PIXEL_MIN,
        "max": PIXEL_MAX,
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

def behavioral(img_in1, img_in2):
    a = img_in1.reshape((HEIGHT, WIDTH)).astype(np.float64)
    b = img_in2.reshape((HEIGHT, WIDTH)).astype(np.float64)
    out = ALPHA * a + BETA * b + GAMMA
    return np.clip(out, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
