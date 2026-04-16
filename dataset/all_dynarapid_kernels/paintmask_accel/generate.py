import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h and testbench) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# INPUT_PTR_WIDTH = 8, OUTPUT_PTR_WIDTH = 8
# color[0] = 150 (hardcoded from testbench: for all channels)

COLOR = 150  # paint color (from testbench: color[i] = 150 for all channels)

function_signature = (
    "void paintmask_accel("
    "int32_t* img_in, "
    "int32_t* mask_in, "
    "int32_t* img_out)"
)

description = f"""\
This HLS kernel paints a fixed color (150) over pixels in a grayscale image
wherever the corresponding mask pixel is non-zero.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}×{WIDTH} grayscale input image in row-major order.
    mask_in: Pointer to a flat array of {HEIGHT * WIDTH} uint8 mask pixel values.
             Non-zero mask values indicate pixels to paint.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 output pixel values.

Operation: For each pixel position (r, c):
    if mask_in[r, c] != 0:
        img_out[r, c] = 150   (hardcoded paint color)
    else:
        img_out[r, c] = img_in[r, c]

The paint color is hardcoded as 150 for all channels (single-channel grayscale image).
Output values are in [0, 255] (no clamping needed since paint color is constant 150).
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
        "name": "mask_in",
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

# Paint color = 150 (hardcoded from testbench)
# if mask != 0 -> output = COLOR, else output = input
def behavioral(img_in, mask_in):
    img = img_in.reshape((128, 128))
    mask = mask_in.reshape((128, 128))
    out = np.where(mask != 0, np.uint8(150), img).astype(np.uint8)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
