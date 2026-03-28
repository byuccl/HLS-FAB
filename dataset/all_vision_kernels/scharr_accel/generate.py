import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

function_signature = (
    "void scharr_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out1, "
    "ap_uint<8>* img_out2)"
)

description = f"""
This HLS kernel applies a Scharr edge detection filter to a grayscale image and computes
horizontal (Gx) and vertical (Gy) gradients.

Parameters:
    img_inp:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
              a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values containing the
              horizontal gradient (Gx) magnitude of the input image.
    img_out2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values containing the
              vertical gradient (Gy) magnitude of the input image.

The kernel computes the horizontal gradient (Gx) and vertical gradient (Gy) using a
3x3 Scharr operator. The Scharr operator is similar to Sobel but with optimized coefficients
for rotational symmetry:

Scharr X (horizontal gradient):
    [[ -3,  0,   3],
     [-10,  0,  10],
     [ -3,  0,   3]]

Scharr Y (vertical gradient):
    [[ -3, -10,  -3],
     [  0,   0,   0],
     [  3,  10,   3]]

Border handling uses zero-padding: pixels outside the image boundary are treated as zero.

For each output pixel:
1. Apply the Scharr X kernel to compute the raw horizontal gradient
2. Apply the Scharr Y kernel to compute the raw vertical gradient  
3. Take the absolute value of each gradient
4. Saturate to [0, 255] and store as uint8

The saturation clamps any values above 255 to 255.
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
        "name": "img_out1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
    {
        "name": "img_out2",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((128, 128))
    gx = cv2.Scharr(img, cv2.CV_16S, 1, 0, borderType=cv2.BORDER_CONSTANT)
    gy = cv2.Scharr(img, cv2.CV_16S, 0, 1, borderType=cv2.BORDER_CONSTANT)
    gx_out = np.clip(np.abs(gx), 0, 255).astype(np.uint8)
    gy_out = np.clip(np.abs(gy), 0, 255).astype(np.uint8)
    return gx_out.flatten(), gy_out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
