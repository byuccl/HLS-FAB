import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# CONVERT_TYPE = XF_CONVERT_8U_TO_16U
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_16UC1
# SHIFT is a runtime parameter; hardcoded here as 0 (no scaling shift)
SHIFT = 0

# XF_8UC1 -> ap_uint<8>, dtype: uint8
# XF_16UC1 -> ap_uint<16>, dtype: uint16
function_signature = (
    "void convert_bitdepth_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<16>* img_out)"
)

description = f"""
This HLS kernel converts a grayscale image from 8-bit unsigned to 16-bit unsigned depth.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values for
             the converted output image (same dimensions as input).

Each output pixel is computed as:
    out[i] = (uint16)(in[i] << SHIFT)

where SHIFT={SHIFT} is a fixed compile-time constant. With SHIFT=0 the output value equals
the input value widened to uint16 (range [0, 255] maps to [0, 255] in uint16). No saturation
is needed since widening conversion cannot overflow.
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
        "dtype": "uint16",
        "iotype": "out",
    },
]

def behavioral(img_in):
    return img_in.astype(np.uint16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
