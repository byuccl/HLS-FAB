import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

function_signature = (
    "void accumulate_squared("
    "ap_uint<8>* img_in1, "
    "ap_uint<8>* img_in2, "
    "ap_uint<16>* img_out)"
)

description = f"""
This HLS kernel accumulates the squared value of one image into another.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the first {HEIGHT}x{WIDTH} grayscale input image in row-major order (image to be squared).
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the second {HEIGHT}x{WIDTH} grayscale input image in row-major order (accumulator).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values for
             the output image (same dimensions as inputs).

The kernel computes:
    img_out[i] = img_in1[i]^2 + img_in2[i]

For each pixel:
1. Square the value from img_in1 (uint8, so result is in range [0, 65025])
2. Add the corresponding value from img_in2 (uint8, range [0, 255])
3. Store the result as uint16 (range [0, 65280])

The uint16 output type prevents overflow. No saturation or clamping is performed.
"""

inputs_info = [
    {
        "name": "img_in1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in2",
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

def behavioral(img_in1, img_in2):
    a = img_in1.astype(np.uint16)
    b = img_in2.astype(np.uint16)
    return (a * a + b).astype(np.uint16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
