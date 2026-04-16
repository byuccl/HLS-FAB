import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameters (not in inputs_info — hardcoded constants)
# These correspond to the runtime float alpha, beta, gamma arguments.
ALPHA = 0.5
BETA  = 0.5
GAMMA = 0.0

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void add_weighted_accel("
    "int32_t* img_in1, "
    "int32_t* img_in2, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel computes a weighted blend of two grayscale images.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
             representing the first {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
             representing the second {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the blended output image (same dimensions as inputs).

Each output pixel is computed as:
    out = clip(alpha * in1 + beta * in2 + gamma, 0, 255)

where alpha={ALPHA}, beta={BETA}, gamma={GAMMA} are fixed compile-time constants.
The intermediate result is computed in float32, then truncated (floor cast) to uint8.
Values below 0 are clamped to 0; values above 255 are clamped to 255.
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
    img1 = img_in1.reshape((128, 128))
    img2 = img_in2.reshape((128, 128))
    out = cv2.addWeighted(img1, 0.5, img2, 0.5, 0.0)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
