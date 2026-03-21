# Example generate.py for: sobel_accel
#
# Top function: sobel_accel
# Source: /home/plwalker/Vitis_Libraries/vision/L1/examples/sobelfilter/
#
# Key decisions:
#   - One input buffer (img_inp) → one "in" entry in inputs_info
#   - Two output buffers (img_out1, img_out2) → two "out" entries in inputs_info
#   - Total: THREE entries in inputs_info
#   - Scalar params (rows, cols) are NOT in inputs_info — they are compile-time constants
#   - inputs_info names match the C++ parameter names exactly
#   - behavioral() returns a TUPLE of two flat arrays (gx, gy)

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from xf_sobel_accel_config.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

function_signature = (
    "void sobel_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out1, "
    "ap_uint<8>* img_out2)"
)

description = f"""
This HLS kernel applies a Sobel edge detection filter to a grayscale image.

Parameters:
    img_inp:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
              representing a {HEIGHT}x{WIDTH} grayscale input image (row-major).
    img_out1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values containing
              the horizontal gradient (Gx) of the input image.
    img_out2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values containing
              the vertical gradient (Gy) of the input image.

The kernel computes the horizontal gradient (Gx) and vertical gradient (Gy) using
a {FILTER_WIDTH}x{FILTER_WIDTH} Sobel operator with the following coefficients:

Sobel X (horizontal gradient):
  [[-1,  0,  1],
   [-2,  0,  2],
   [-1,  0,  1]]

Sobel Y (vertical gradient):
  [[-1, -2, -1],
   [ 0,  0,  0],
   [ 1,  2,  1]]

Border handling: Zero-padding is used for pixels outside the image boundary.

Division/rounding: The convolution result is computed as int16, then saturated to
the range [0, 255] and cast to uint8. Negative values are clamped to 0, values above
255 are clamped to 255.

Output: Both img_out1 (Gx) and img_out2 (Gy) are uint8 images with the same dimensions
as the input ({HEIGHT}x{WIDTH}).
"""

# inputs_info has THREE entries: one per pointer parameter in function_signature.
# Names match the C++ parameter names exactly.
# behavioral() returns a TUPLE (gx_array, gy_array) — one element per "out" entry.
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
        "name": "img_out1",          # Gx (horizontal gradient)
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
    {
        "name": "img_out2",          # Gy (vertical gradient)
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]

# Standard 3x3 Sobel kernels
_SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
_SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

def _convolve2d(img, kernel):
    """2-D convolution with zero-padding."""
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return out

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.float32)
    gx = _convolve2d(img, _SOBEL_X)
    gy = _convolve2d(img, _SOBEL_Y)
    # Saturate to [0, 255] and cast to uint8
    gx_uint8 = np.clip(gx, 0, 255).astype(np.uint8)
    gy_uint8 = np.clip(gy, 0, 255).astype(np.uint8)
    # Return a TUPLE: one flat array per "out" entry in inputs_info (in order).
    return gx_uint8.flatten(), gy_uint8.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
