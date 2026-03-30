import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

# Hardcoded sigma values used by the kernel
SIGMA_COLOR = 10.0
SIGMA_SPACE = 0.5

function_signature = (
    "void bilateral_filter_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel applies a bilateral filter to a grayscale image.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The bilateral filter smooths the image while preserving edges. It uses a {FILTER_WIDTH}x{FILTER_WIDTH}
circular kernel (cross-shaped for 3x3: center + 4 cardinal neighbors), sigma_color={SIGMA_COLOR},
sigma_space={SIGMA_SPACE}.

For each output pixel:
1. Apply replicate border padding (1 pixel on all sides).
2. Accumulate over the 5-pixel cross (center + 4 cardinals):
   - Center pixel: weight = 1.0
   - Cardinal neighbors: weight = quantized LUT value (ap_ufixed<16,1>)
     where LUT[m] = exp(-0.5 * m^2 / sigma_color^2) * exp(-0.5 * 1.0 / sigma_space^2),
     and m = |pixel_value_difference|
3. Output = round(px_sum / weight_sum) as uint8.
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
    out = cv2.bilateralFilter(img, 3, 10.0, 0.5, borderType=cv2.BORDER_REPLICATE)
    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
