# Example generate.py for: add_weighted_accel
#
# Top function: add_weighted_accel
# Source: /home/plwalker/Vitis_Libraries/vision/L1/examples/addweighted/
#
# Key decisions:
#   - Two image input buffers (img_in1, img_in2) → two "in" entries in inputs_info.
#   - One image output buffer (img_out) → one "out" entry in inputs_info.
#   - Total: THREE entries in inputs_info.
#   - Scalar params (alpha, beta, gamma, height, width) are NOT in inputs_info.
#     They are hardcoded as module-level constants.
#   - inputs_info names match the C++ parameter names exactly.
#   - The function_signature lists pointer params only (no scalars, no height/width).

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameters (not in inputs_info — hardcoded constants)
ALPHA = 0.5
BETA  = 0.5
GAMMA = 0.0

function_signature = (
    "void add_weighted_accel("
    "uint32_t* img_in1, "
    "uint32_t* img_in2, "
    "uint32_t* img_out)"
)

description = f"""
This HLS kernel computes a weighted blend of two grayscale images.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint32 values representing 8bit pixel values
             representing the first {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint32 values representing 8bit pixel values
             representing the second {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint32 values representing 8bit pixel values for
             the blended output image (same dimensions as inputs).

Each output pixel is computed as:
    out = clip(alpha * in1 + beta * in2 + gamma, 0, 255)

where alpha={ALPHA}, beta={BETA}, gamma={GAMMA} are fixed compile-time constants.
The intermediate result is computed in float32, then truncated (floor cast) to uint32.
Values below 0 are clamped to 0; values above 255 are clamped to 255.
"""

# inputs_info has THREE entries: img_in1 (in), img_in2 (in), img_out (out).
# The number of entries equals the number of pointer parameters in function_signature.
inputs_info = [
    {
        "name": "img_in1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in2",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint23",
        "iotype": "out",
    },
]

def behavioral(img_in1, img_in2):
    # behavioral() receives one argument per "in" entry in inputs_info, in order.
    a = img_in1.reshape((HEIGHT, WIDTH)).astype(np.float32)
    b = img_in2.reshape((HEIGHT, WIDTH)).astype(np.float32)
    # Weighted blend: alpha*a + beta*b + gamma, clamped to [0, 255], truncated to uint8
    out = np.clip(ALPHA * a + BETA * b + GAMMA, 0, 255).astype(np.uint32)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
