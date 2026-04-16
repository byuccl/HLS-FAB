import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
NEWHEIGHT = 64
NEWWIDTH = 64
# INTERPOLATION = 1 (bilinear), MAXDOWNSCALE = 2

# XF_8UC1 -> ap_uint<8>, dtype: uint8
# Input: HEIGHT x WIDTH, Output: NEWHEIGHT x NEWWIDTH
function_signature = (
    "void resize_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel resizes a grayscale image from {HEIGHT}x{WIDTH} to {NEWHEIGHT}x{NEWWIDTH}
using bilinear interpolation.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {NEWHEIGHT * NEWWIDTH} uint8 pixel values for
             the resized output image ({NEWHEIGHT}x{NEWWIDTH}, row-major order).

For each output pixel at position (r, c), the corresponding input coordinates use
center-aligned mapping:
    x_in = (c + 0.5) * ({WIDTH} / {NEWWIDTH}) - 0.5  =  2*c + 0.5
    y_in = (r + 0.5) * ({HEIGHT} / {NEWHEIGHT}) - 0.5  =  2*r + 0.5

The four nearest input pixels at (floor(y_in), floor(x_in)) to (ceil(y_in), ceil(x_in))
are combined using bilinear interpolation with fractional weights (Wx, Wy):
    result = A0 + (B0-A0)*Wx + (A1-A0)*Wy + (A0+B1-B0-A1)*Wx*Wy
where A0=img[y0,x0], B0=img[y0,x1], A1=img[y1,x0], B1=img[y1,x1].
The result is truncated (floor) to uint8. For this {HEIGHT}x{WIDTH} -> {NEWHEIGHT}x{NEWWIDTH}
configuration all fractional coordinates are exactly 0.5, giving:
    result = floor((A0 + B0 + A1 + B1) / 4)
When the interpolation window extends to the last row or column, the edge pixel is repeated.
"""

inputs_info = [
    {
        "name": "img_inp",
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
        "shape": (NEWHEIGHT, NEWWIDTH),
        "total_elements": NEWHEIGHT * NEWWIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((128, 128))
    out = cv2.resize(img, (64, 64), interpolation=cv2.INTER_LINEAR)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
