# Example generate.py for: boxfilter (DynaRapid int32 version)
#
# Vitis Vision kernel: ${VITIS_LIBRARIES}/vision/L1/examples/boxfilter/
#
# Source files read to produce this:
#   - xf_box_filter_accel.cpp        -> top function name: boxfilter_accel
#   - config/xf_config_params.h      -> HEIGHT=128, WIDTH=128, IN_TYPE=XF_8UC1,
#                                       FILTER_WIDTH=3
#   - description.json               -> topfunction: boxfilter_accel
#
# DynaRapid adaptation:
#   - Original XF_8UC1 (uint8, range 0-255) → int32_t (range 0-255 in int32 container)
#   - All pointer params → int32_t*
#   - behavioral() clamps output to [0, 255]
#   - Function name: boxfilter_accel_int32
#
# C++ source analysis:
#   - Division method: mean of FILTER_WIDTH^2 window, truncated to integer
#   - Border handling: zero-padding (XF_BORDER_CONSTANT), border pixels divided
#     by full FILTER_WIDTH^2 (9), not just in-bounds pixel count
#   - Accumulator: int32 (sum of up to 9 uint8 values, max 2295, fits int32)
#   - Output saturation: truncated cast to uint8 range [0, 255]

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

# --- Logical pixel range ---
# Original Vitis type: XF_8UC1 → logical range [0, 255]
# Values are stored in int32 but represent uint8 pixels.
PIXEL_MIN = 0
PIXEL_MAX = 255

function_signature = (
    "void boxfilter_accel_int32("
    "int32_t* img_inp, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel applies a box filter (mean filter) to a grayscale image.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} int32 values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
             Each value represents a pixel in the range [{PIXEL_MIN}, {PIXEL_MAX}].
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} int32 values for the
             filtered output image (same dimensions as input).
             Each output value is clamped to [{PIXEL_MIN}, {PIXEL_MAX}].

The kernel applies a {FILTER_WIDTH}x{FILTER_WIDTH} box filter (uniform mean filter) to each
pixel. For each output pixel, the kernel computes the sum of all pixels in the
{FILTER_WIDTH}x{FILTER_WIDTH} neighbourhood and divides by {FILTER_WIDTH * FILTER_WIDTH}
using integer division (truncation toward zero).

Border handling: pixels outside the image boundary are treated as zero (zero-padding).
Border pixels are still divided by the full window size ({FILTER_WIDTH * FILTER_WIDTH}),
not by the count of in-bounds pixels only.

Pixel values are logically 8-bit unsigned (0-255) stored in int32. The output is
clamped to [0, 255].
"""

inputs_info = [
    {
        "name": "img_inp",
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

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.int64)
    pad = FILTER_WIDTH // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            window_sum = padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH].sum()
            out[i, j] = window_sum // (FILTER_WIDTH * FILTER_WIDTH)
    return np.clip(out, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
