import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

function_signature = (
    "void accumulate_accel("
    "int32_t* img_in1, "
    "int32_t* img_in2, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel accumulates (adds) two grayscale images element-wise.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the first {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the second {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values for
             the output image (same dimensions as inputs).

The kernel computes the element-wise sum of two uint8 images and stores the result
as uint16 to prevent overflow:
    out[i] = img_in1[i] + img_in2[i]

Each output pixel is the direct sum of the corresponding input pixels, with no saturation
or clamping. The uint16 output type can represent values from 0 to 510 (2*255).
"""

inputs_info = [
    {
        "name": "img_in1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in2",
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

def behavioral(img_in1, img_in2):
    a = img_in1.astype(np.uint16)
    b = img_in2.astype(np.uint16)
    return (a + b).astype(np.uint16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
