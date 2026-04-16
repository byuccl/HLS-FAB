import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# CRITICAL: Dynarapid ONLY accepts int32_t for all pointer parameters.
# All Vitis types convert to int32_t. Float types (XF_32FC1, ap_fixed, ap_ufixed, float, double) must be SKIPPED.
function_signature = "void accumulate_accel(int32_t* img_in1, int32_t* img_in2, int32_t* img_out)"

description = f"""
This HLS kernel accumulates two grayscale images by pixel-wise addition.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values for
             the output image (same dimensions as input).

This kernel performs element-wise addition of the two input images:
    out[i,j] = in1[i,j] + in2[i,j]

Each pixel is converted from uint8 (0-255) to uint16 before addition. The maximum output
value is 255 + 255 = 510, which fits in uint16 without saturation or clamping.
"""

# inputs_info MUST have one entry per pointer parameter in function_signature,
# with the following rules:
#
#   RULE 1 — one input, one output (most common):
#     Two entries: one "in", one "out".
#     behavioral() takes one argument, returns one flat array.
#
#   RULE 2 — multiple inputs, one output (e.g. addweighted: img_in1, img_in2, img_out):
#     Three or more entries: multiple "in", one "out".
#     behavioral() takes one argument per "in" entry, in order.
#
#   RULE 3 — one input, multiple outputs (e.g. sobelfilter: img_inp, img_out1, img_out2):
#     PromptConstructor requires exactly ONE "out" entry.
#     Merge all output buffers into a single combined entry (e.g. shape (2, H, W)).
#     behavioral() returns a single flat concatenated array.
#     The merged entry's "name" is a logical label — it does NOT need to match
#     any C++ parameter name.
#
# Count entries by counting pointer (int32_t*) parameters in function_signature,
# then apply RULE 3 if there are multiple outputs.

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
    """
    Implement pixel-wise accumulation (addition) of two input images.

    The C++ implementation reads 8-bit pixels from both input images, casts them to 16-bit,
    adds them, and stores the result as 16-bit unsigned. There is no saturation or clamping
    because the maximum sum (255 + 255 = 510) fits in uint16 (max 65535).
    """
    # Reshape inputs from flat arrays to 2D images
    img1 = img_in1.reshape((HEIGHT, WIDTH)).astype(np.uint16)
    img2 = img_in2.reshape((HEIGHT, WIDTH)).astype(np.uint16)

    # Accumulate: element-wise addition
    result = img1 + img2

    # Return as flat uint16 array
    return result.astype(np.uint16).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)
pc.write_json("kernel.json")
