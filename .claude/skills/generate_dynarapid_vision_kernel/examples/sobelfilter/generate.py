# Example generate.py for: sobelfilter (DynaRapid int32 version)
#
# Vitis Vision kernel: ${VITIS_LIBRARIES}/vision/L1/examples/sobelfilter/
#
# Source files read to produce this:
#   - xf_sobel_accel.cpp              -> top function name: sobel_accel
#   - config/xf_config_params.h       -> HEIGHT=128, WIDTH=128, IN_TYPE=XF_8UC1,
#                                        OUT_TYPE=XF_16SC1, FILTER_WIDTH=3
#   - description.json                -> topfunction: sobel_accel
#
# DynaRapid adaptation:
#   - Input: XF_8UC1 (uint8, range 0-255) → int32_t (range 0-255 in int32 container)
#   - Output: XF_16SC1 (int16, range -32768 to 32767) → int32_t (same range in int32)
#   - All pointer params → int32_t*
#   - behavioral() clamps Gx/Gy to [-32768, 32767]
#   - Function name: sobel_accel_int32
#
# This kernel demonstrates:
#   - Different input/output logical ranges (uint8 input, int16 output)
#   - Multiple outputs (Gx and Gy) returned as a tuple
#   - Manual 2D convolution with Sobel kernels
#
# C++ source analysis:
#   - Coefficients: standard 3x3 Sobel ([-1,0,1; -2,0,2; -1,0,1] for Gx)
#   - Border handling: zero-padding
#   - Output: int16 range, no saturation needed for 3x3 on uint8 input
#     (max possible value: 4*255 = 1020, fits in int16)

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

# --- Logical pixel ranges ---
# Input: XF_8UC1 → logical range [0, 255]
# Output: XF_16SC1 → logical range [-32768, 32767]
INPUT_PIXEL_MIN = 0
INPUT_PIXEL_MAX = 255
OUTPUT_PIXEL_MIN = -32768
OUTPUT_PIXEL_MAX = 32767

function_signature = (
    "void sobel_accel_int32("
    "int32_t* img_inp, "
    "int32_t* img_out1, "
    "int32_t* img_out2)"
)

description = f"""
This HLS kernel applies a Sobel edge detection filter to a grayscale image.

Parameters:
    img_inp:  Pointer to a flat array of {HEIGHT * WIDTH} int32 values representing
              a {HEIGHT}x{WIDTH} grayscale input image (row-major).
              Each value represents a pixel in the range [{INPUT_PIXEL_MIN}, {INPUT_PIXEL_MAX}].
    img_out1: Pointer to a flat array of {HEIGHT * WIDTH} int32 values containing
              the horizontal gradient (Gx) of the input image.
              Each value is in the range [{OUTPUT_PIXEL_MIN}, {OUTPUT_PIXEL_MAX}].
    img_out2: Pointer to a flat array of {HEIGHT * WIDTH} int32 values containing
              the vertical gradient (Gy) of the input image.
              Each value is in the range [{OUTPUT_PIXEL_MIN}, {OUTPUT_PIXEL_MAX}].

The kernel computes the horizontal gradient (Gx) and vertical gradient (Gy) using
3x3 Sobel operators:

    Gx kernel: [[-1, 0, 1],    Gy kernel: [[-1, -2, -1],
                [-2, 0, 2],                [ 0,  0,  0],
                [-1, 0, 1]]                [ 1,  2,  1]]

Out-of-bounds pixels are treated as zero (zero-padding). Input pixel values are
logically 8-bit unsigned (0-255) stored in int32. Output gradient values are
logically 16-bit signed (-32768 to 32767) stored in int32.
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": INPUT_PIXEL_MIN,
        "max": INPUT_PIXEL_MAX,
    },
    {
        "name": "img_out1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "img_out2",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

# Standard 3x3 Sobel kernels
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int64)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int64)

def _convolve2d(img, kernel):
    """2-D convolution with zero-padding, integer arithmetic."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return out

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.int64)
    gx = _convolve2d(img, _SOBEL_X)
    gy = _convolve2d(img, _SOBEL_Y)
    gx_out = np.clip(gx, OUTPUT_PIXEL_MIN, OUTPUT_PIXEL_MAX).astype(np.int32)
    gy_out = np.clip(gy, OUTPUT_PIXEL_MIN, OUTPUT_PIXEL_MAX).astype(np.int32)
    return gx_out.flatten(), gy_out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
