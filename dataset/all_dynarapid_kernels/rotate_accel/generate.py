import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 512
WIDTH = 512

# Rotation direction: 0=180deg, 1=90deg clockwise, 2=270deg clockwise
# R_90=1 means 90-degree clockwise rotation
DIRECTION = 1  # 90-degree clockwise

function_signature = (
    "void rotate_accel("
    "int32_t* img_in, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel rotates a grayscale image by 90 degrees clockwise.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {WIDTH * HEIGHT} uint8 pixel values for
             the output image ({WIDTH}x{HEIGHT}, dimensions swapped after 90-degree rotation).

The rotation direction is {DIRECTION} (90 degrees clockwise).

For 90-degree clockwise rotation:
- The output image has dimensions {WIDTH}x{HEIGHT} (height and width are swapped)
- Output pixel at (i, j) = Input pixel at (HEIGHT - 1 - j, i)
- The first row of the input becomes the last column of the output
- The last row of the input becomes the first column of the output

For other rotation directions:
- direction = 0: 180-degree rotation (dimensions unchanged)
  Output (i, j) = Input (HEIGHT - 1 - i, WIDTH - 1 - j)
- direction = 1: 90-degree clockwise (dimensions swapped)
  Output (i, j) = Input (HEIGHT - 1 - j, i)
- direction = 2: 270-degree clockwise = 90-degree counter-clockwise (dimensions swapped)
  Output (i, j) = Input (j, WIDTH - 1 - i)
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
        "shape": (WIDTH, HEIGHT),  # Dimensions swapped for 90-degree rotation
        "total_elements": WIDTH * HEIGHT,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((512, 512))
    out = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
