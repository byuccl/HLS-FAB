import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameter
ALPHA = 0.76

function_signature = (
    "void accumulate_weighted_accel("
    "ap_uint<8>* img_in1, "
    "ap_uint<8>* img_in2, "
    "ap_uint<16>* img_out)"
)

description = f"""
This HLS kernel computes a weighted accumulation of two grayscale images using Q0.23
fixed-point arithmetic, matching the Xilinx xf::cv::accumulateWeighted implementation.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the first {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             the second {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values for
             the output image (same dimensions as inputs).

The kernel uses Q0.23 fixed-point arithmetic as follows (alpha = {ALPHA}):
    alpha_fp  = (uint32_t)(alpha * ((2^23) - 1))   // truncated via ap_uint<24> cast
    beta_fp   = ((2^23) - 1) - alpha_fp + 1
    img_out[i] = (img_in1[i] * alpha_fp + img_in2[i] * beta_fp) >> 23

Both multiplications use 40-bit unsigned intermediate products before summing and
right-shifting by 23. The result is stored as uint16 (no saturation needed since
inputs are uint8 and weights sum to approximately 2^23).
"""

inputs_info = [
    {
        "name": "img_in1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in2",
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
        "dtype": "uint16",
        "iotype": "out",
    },
]

def behavioral(img_in1, img_in2):
    # Exact Q0.23 fixed-point arithmetic matching xf::cv::accumulateWeighted
    # ap_uint<24> temp = (ap_uint<24>)(alpha * ((1<<23) - 1))
    alpha_fp = np.uint32(np.float32(ALPHA) * np.float32((1 << 23) - 1))
    beta_fp  = np.uint32(((1 << 23) - 1) - alpha_fp + 1)
    a = img_in1.astype(np.uint64)
    b = img_in2.astype(np.uint64)
    result = (a * np.uint64(alpha_fp) + b * np.uint64(beta_fp)) >> np.uint64(23)
    return result.astype(np.uint16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
