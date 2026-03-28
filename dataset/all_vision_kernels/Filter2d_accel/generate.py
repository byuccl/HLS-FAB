import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h and testbench) ---
HEIGHT = 128
WIDTH = 128
FILTER_HEIGHT = 3
FILTER_WIDTH = 3
SHIFT = 15
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# OUT_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# INPUT_PTR_WIDTH = 8, OUTPUT_PTR_WIDTH = 8
# Border: XF_BORDER_CONSTANT (zero padding)
# Filter coefficients: all 3640 (short int), from testbench: filter_ptr[i*3+j] = 3640
# Shift: 15 (SHIFT), from config

FILTER_COEFF = 3640  # all 9 elements = 3640 (short int)

function_signature = (
    "void Filter2d_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""\
This HLS kernel applies a custom {FILTER_HEIGHT}×{FILTER_WIDTH} convolution filter to a
{HEIGHT}×{WIDTH} grayscale image using zero-padding at borders.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}×{WIDTH} grayscale image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 output pixel values.

Filter coefficients (hardcoded): all {FILTER_HEIGHT * FILTER_WIDTH} = {FILTER_COEFF} (short int values).
Shift parameter (hardcoded): {SHIFT}.

Operation for each output pixel (r, c):
    acc = sum of (pixel(r+dr-1, c+dc-1) × {FILTER_COEFF}) for dr,dc in 0..2
          (pixels outside image boundary treated as 0 — zero-padding)
    result = acc >> {SHIFT}  (arithmetic right-shift / floor division by {2**SHIFT})
    img_out[r, c] = saturate(result, 0, 255)  (clamp to uint8 range)

The filter approximates an averaging filter: sum of coefficients = {FILTER_HEIGHT * FILTER_WIDTH * FILTER_COEFF},
divided by {2**SHIFT} = {FILTER_HEIGHT * FILTER_WIDTH * FILTER_COEFF / (2**SHIFT):.6f} ≈ 1.0 per pixel.
Output is stored as uint8 (values saturated to [0, 255]).
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

# Division: arithmetic right-shift by SHIFT=15 (floor for positive values)
# Border: zero-padding (XF_BORDER_CONSTANT with value 0)
# Accumulator: int32 (sufficient: max = 9 * 255 * 3640 = 8,354,400 < 2^31)
# Output: uint8, saturated to [0, 255]
def behavioral(img_in):
    img = img_in.reshape((128, 128))
    kernel = np.ones((3, 3), dtype=np.float64) * (3640.0 / 32768.0)
    out = cv2.filter2D(img, -1, kernel, borderType=cv2.BORDER_CONSTANT)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
