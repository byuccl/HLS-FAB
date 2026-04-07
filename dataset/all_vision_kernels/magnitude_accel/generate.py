import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Norm type: XF_L1NORM (L1 norm = |x| + |y|)
NORM_TYPE = "L1"

function_signature = (
    "void magnitude_accel("
    "ap_uint<16>* img_inp1, "
    "ap_uint<16>* img_inp2, "
    "ap_uint<16>* img_out)"
)

description = f"""
This HLS kernel computes the magnitude of 2D gradient vectors using the L1 norm.

Parameters:
    img_inp1: Pointer to a flat array of {HEIGHT * WIDTH} int16 values representing
              the x-component (horizontal gradient) of a {HEIGHT}x{WIDTH} gradient field in row-major order.
    img_inp2: Pointer to a flat array of {HEIGHT * WIDTH} int16 values representing
              the y-component (vertical gradient) of a {HEIGHT}x{WIDTH} gradient field in row-major order.
    img_out:  Pointer to a flat array of {HEIGHT * WIDTH} int16 values for
              the output magnitude image (same dimensions as inputs).

The kernel computes the L1 norm (Manhattan distance) of each gradient vector:
    mag[i] = |gx[i]| + |gy[i]|

For each pixel:
1. Take the absolute value of the x-component (img_inp1[i])
2. Take the absolute value of the y-component (img_inp2[i])
3. Add the two absolute values
4. Store the result as int16

The result is stored as int16. For typical Sobel gradients (range approximately [-1020, 1020]),
the L1 magnitude can be up to 2040.
"""

inputs_info = [
    {
        "name": "img_inp1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int16",
        "iotype": "in",
        "min": -1020,
        "max": 1020,
    },
    {
        "name": "img_inp2",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int16",
        "iotype": "in",
        "min": -1020,
        "max": 1020,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int16",
        "iotype": "out",
    },
]

def behavioral(img_inp1, img_inp2):
    gx = img_inp1.reshape((128, 128)).astype(np.int32)
    gy = img_inp2.reshape((128, 128)).astype(np.int32)
    return (np.abs(gx) + np.abs(gy)).astype(np.int16).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
