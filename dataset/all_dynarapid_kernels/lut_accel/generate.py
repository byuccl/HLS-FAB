import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# lut_ptr is an unsigned char[256] input buffer (not ap_uint<N>*).
# It is hardcoded here as an identity LUT (output[i] = i, no transformation).
LUT = np.arange(256, dtype=np.uint8)

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void lut_accel("
    "int32_t* img_in, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel applies a 256-entry lookup table (LUT) to each pixel of a grayscale image.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the transformed output image (same dimensions as input).

Each output pixel is computed as:
    out[i] = LUT[in[i]]

where LUT is a fixed 256-element lookup table indexed by the input pixel value (0-255).
The LUT used here is the identity mapping: LUT[v] = v (output equals input).
The transformation is applied independently to every pixel with no spatial context.
Output values are uint8 in the range [0, 255].
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
    lut = np.arange(256, dtype=np.uint8)
    out = cv2.LUT(img, lut)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
