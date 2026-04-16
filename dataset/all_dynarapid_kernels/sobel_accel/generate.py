import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

function_signature = (
    "void sobel_accel("
    "int32_t* img_inp, "
    "int32_t* img_out1, "
    "int32_t* img_out2)"
)

description = f"""
This HLS kernel applies a Sobel edge detection filter to a grayscale image and computes
horizontal (Gx) and vertical (Gy) gradients.

Parameters:
    img_inp:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
              a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values containing the
              horizontal gradient (Gx) magnitude of the input image.
    img_out2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values containing the
              vertical gradient (Gy) magnitude of the input image.

The kernel computes the horizontal gradient (Gx) and vertical gradient (Gy) using a
{FILTER_WIDTH}x{FILTER_WIDTH} Sobel operator with the following kernels:

Sobel X (horizontal gradient):
    [[-1,  0,  1],
     [-2,  0,  2],
     [-1,  0,  1]]

Sobel Y (vertical gradient):
    [[-1, -2, -1],
     [ 0,  0,  0],
     [ 1,  2,  1]]

Border handling uses zero-padding: pixels outside the image boundary are treated as zero.

For each output pixel:
1. Apply the Sobel X kernel to compute the raw horizontal gradient (range: -1020 to 1020)
2. Apply the Sobel Y kernel to compute the raw vertical gradient (range: -1020 to 1020)
3. Clamp each signed gradient to [0, 255]: negative values become 0, values above 255 become 255
4. Store as uint8

Note: negative gradient values are clamped to 0 (NOT absolute value). Only the positive
portion of each gradient is retained.
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "img_out2",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((128, 128))
    gx = cv2.Sobel(img, cv2.CV_16S, 1, 0, ksize=3, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.Sobel(img, cv2.CV_16S, 0, 1, ksize=3, borderType=cv2.BORDER_CONSTANT)
    gx_out = np.clip(gx, 0, 255).astype(np.uint8)
    gy_out = np.clip(gy, 0, 255).astype(np.uint8)
    return gx_out.flatten(), gy_out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
