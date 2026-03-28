import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH_1 = 3   # first Gaussian filter size (FILTER_SIZE_1_3 = 1)
FILTER_WIDTH_2 = 3   # second Gaussian filter size (FILTER_SIZE_2_3 = 1)
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# sigma1 and sigma2 are runtime float parameters; hardcoded here.
SIGMA1 = 1.0
SIGMA2 = 2.0

# Algorithm (from accel source):
#   imgin1 = GaussianBlur(imgInput, FILTER_WIDTH_1, sigma1)  -- with zero-padding
#   (imgin3 = duplicate of imgin1)
#   imgin4 = GaussianBlur(imgin1, FILTER_WIDTH_2, sigma2)    -- second Gaussian on first result
#   imgOutput = subtract_saturate(imgin3, imgin4)            -- XF_CONVERT_POLICY_SATURATE
#
# Division method: Gaussian coefficients are float (sigma-dependent), normalized to sum=1.
#                  Convolution result is float; cast to uint8 truncates.
# Border: zero-padding (XF_BORDER_CONSTANT); border pixels include zero-valued out-of-bounds.
# Subtraction: saturated to [0, 255] (uint8 output cannot go below 0).

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void gaussian_diff_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel computes a Difference of Gaussians (DoG) on a grayscale image.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the DoG output image (same dimensions as input).

The kernel applies the following steps:

1. Apply a {FILTER_WIDTH_1}x{FILTER_WIDTH_1} Gaussian blur with sigma={SIGMA1} to img_in
   using zero-padding at borders. The 1D separable kernel weights are computed from
   float32 Gaussian values, normalized, multiplied by 256, and rounded to uint8 with
   error diffusion accumulation. The 2D kernel is the outer product of the 1D weights.
   Convolution: result = (sum(K[i,j] * pixel[i,j]) + 32768) >> 16. Output stored as int32
   (call this G1).

2. Apply a second {FILTER_WIDTH_2}x{FILTER_WIDTH_2} Gaussian blur with sigma={SIGMA2} to
   G1 using zero-padding. Same integer convolution method. Output stored as int32 (G2).

3. Compute the difference: out = saturate(G1 - G2, 0, 255).
   Subtraction is performed in integer. The result is clamped to [0, 255] and cast to uint8.
   Negative values become 0; values above 255 become 255.

sigma1={SIGMA1}, sigma2={SIGMA2} are fixed compile-time constants.
"""

inputs_info = [
    {
        "name": "img_in",
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
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((128, 128))
    g1 = cv2.GaussianBlur(img, (3, 3), 1.0, borderType=cv2.BORDER_CONSTANT)
    g2 = cv2.GaussianBlur(g1, (3, 3), 2.0, borderType=cv2.BORDER_CONSTANT)
    diff = g1.astype(np.int16) - g2.astype(np.int16)
    return np.clip(diff, 0, 255).astype(np.uint8).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
