import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# OUTPUT: double* sum_out (1 element) -> modeled as ap_uint<64>*, dtype: float64
# INPUT_PTR_WIDTH = 8
# GRAY = 1 (single channel)

function_signature = (
    "void sum_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<64>* sum_out)"
)

description = f"""\
This HLS kernel computes the sum of all pixel values in a grayscale image and
returns it as a double-precision floating-point value.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}×{WIDTH} grayscale input image in row-major order.
    sum_out: Pointer to a single 64-bit value (double) holding the pixel sum.
             The value is the exact integer sum of all {HEIGHT * WIDTH} pixel values,
             cast to double (no rounding error since the sum fits exactly in a 64-bit
             double mantissa: maximum value = 255 × {HEIGHT * WIDTH} = {255 * HEIGHT * WIDTH}).

Operation: Accumulates all pixel values into a 64-bit integer accumulator, then stores
the result as a double. The output ap_uint<64> holds the IEEE 754 double bit pattern.
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
        "name": "sum_out",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "float64",
        "iotype": "out",
    },
]

def behavioral(img_in):
    return np.array([np.sum(img_in.astype(np.float64))], dtype=np.float64)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
