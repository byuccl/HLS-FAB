import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 1080
WIDTH = 1920

# Bayer pattern: BG (value 0)
# For BG: even row/even col -> R(0), even row/odd col -> G(1),
#          odd row/even col -> G(1), odd row/odd col -> B(2)
# All 3 channels have identical params, so color_idx doesn't affect output.
BAYERP = 0

# Degamma piecewise-linear params: 8 segments per channel, same for all 3 channels.
# Each segment: [upper_bound, slope_q14, intercept]
# slope_q14 = floor(float_slope * 16384)
# The float params (from testbench): {upper_bound, slope, intercept}
#   {32, 0.08, 0}, {64, 0.3, 7}, {96, 0.55, 23}, {128, 0.82, 49},
#   {160, 1.1, 84}, {192, 1.4, 132}, {224, 1.75, 200}, {256, 2.0, 256}
DEGAMMA_PARAMS = np.array([
    [32, 1310, 0],
    [64, 4915, 7],
    [96, 9011, 23],
    [128, 13434, 49],
    [160, 18022, 84],
    [192, 22937, 132],
    [224, 28672, 200],
    [256, 32768, 256],
], dtype=np.int64)

function_signature = (
    "void degamma_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""Degamma correction on a {HEIGHT}x{WIDTH} single-channel 8-bit Bayer-pattern image
using a piecewise linear transfer function.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
             representing a {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 output pixel values.

The degamma curve is defined by 8 piecewise linear segments. Each segment has an
upper bound, a fixed-point slope (in Q18.14 format), and an integer intercept.

For each pixel, the segment is selected by finding the first segment whose upper bound
exceeds the pixel value. The output is computed as:
    out = ((slope_q14 * pixel) >> 14) - intercept

The result is stored as uint8 (lower 8 bits).

Segment table (same for all Bayer positions):
    Upper bound | Slope (Q18.14) | Intercept
    32          | 1310           | 0
    64          | 4915           | 7
    96          | 9011           | 23
    128         | 13434          | 49
    160         | 18022          | 84
    192         | 22937          | 132
    224         | 28672          | 200
    256         | 32768          | 256
"""

inputs_info = [
    {
        "name": "img_in",
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


def behavioral(img_in):
    pixels = img_in.astype(np.int64).flatten()
    out = np.zeros_like(pixels)

    # All channels have same params, so no need to distinguish Bayer position.
    # Process each segment: find first segment where pixel < upper_bound
    remaining = np.ones(len(pixels), dtype=bool)
    for seg_idx in range(8):
        ub = DEGAMMA_PARAMS[seg_idx, 0]
        slope = DEGAMMA_PARAMS[seg_idx, 1]
        intercept = DEGAMMA_PARAMS[seg_idx, 2]

        mask = remaining & (pixels < ub)
        out[mask] = ((slope * pixels[mask]) >> 14) - intercept
        remaining &= ~mask

    return (out & 0xFF).astype(np.uint8)


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
