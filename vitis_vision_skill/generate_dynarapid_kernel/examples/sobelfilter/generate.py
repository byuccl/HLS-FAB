# Example generate.py for: sobel_accel
#
# Top function: sobel_accel
# Source: /home/plwalker/Vitis_Libraries/vision/L1/examples/sobelfilter/
#
# C++ function signature (from xf_config_params.h):
#   void sobel_accel(ap_uint<8>* img_inp,      <- pointer -> inputs_info "in" entry
#                    ap_uint<8>* img_out1,      <- pointer -> would be "out" entry 1
#                    ap_uint<8>* img_out2,      <- pointer -> would be "out" entry 2
#                    int rows,                  <- scalar  -> NOT in inputs_info
#                    int cols)                  <- scalar  -> NOT in inputs_info
#
# inputs_info entry count:
#   - 3 pointer params -> would normally mean 3 inputs_info entries.
#   - BUT PromptConstructor requires EXACTLY ONE "out" entry.
#   - img_out1 and img_out2 are BOTH outputs -> they MUST be merged into one entry.
#   - Result: 2 inputs_info entries total (1 "in" + 1 merged "out").
#   - The merged "out" entry has shape (2, HEIGHT, WIDTH) and a descriptive name.
#   - The merged name does NOT need to match any C++ parameter name.
#
# behavioral():
#   - Returns np.concatenate([gx.flatten(), gy.flatten()]) as a single flat array.
#   - First HEIGHT*WIDTH elements = img_out1 (Gx), next HEIGHT*WIDTH = img_out2 (Gy).

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

function_signature = (
    "void sobel_accel("
    "uint32_t* img_inp, "
    "int32_t* img_out1, "
    "int32_t* img_out2)"
)

description = f"""
This HLS kernel applies a Sobel edge detection filter to a grayscale image.

Parameters:
    img_inp:  Pointer to a flat array of {HEIGHT * WIDTH} uint32 pixel values
              representing a {HEIGHT}x{WIDTH} grayscale input image (row-major). Range from 0-255
    img_out1: Pointer to a flat array of {HEIGHT * WIDTH} int32 values containing
              the horizontal gradient (Gx) of the input image. 
    img_out2: Pointer to a flat array of {HEIGHT * WIDTH} int32 values containing
              the vertical gradient (Gy) of the input image.

The kernel computes the horizontal gradient (Gx) and vertical gradient (Gy) of the
input image using a {FILTER_WIDTH}x{FILTER_WIDTH} Sobel operator. Out-of-bounds samples
are treated as zero (zero-padding). img_out1 and img_out2 are both int16 images
of the same dimensions as the input.

Note: In this generate.py, the two outputs are concatenated into a single flat array
of length {HEIGHT * WIDTH * 2}: the first {HEIGHT * WIDTH} elements are Gx (img_out1)
and the next {HEIGHT * WIDTH} elements are Gy (img_out2).
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
        "dtype": "int16",
        "iotype": "out",
    },
    {
        "name": "img_out2",          # Gy (vertical gradient)
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int16",
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
    gx_int16 = np.clip(gx, -32768, 32767).astype(np.int32)
    gy_int16 = np.clip(gy, -32768, 32767).astype(np.int32)
    # Return a TUPLE: one flat array per "out" entry in inputs_info (in order).
    return gx_int16.flatten(), gy_int16.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
