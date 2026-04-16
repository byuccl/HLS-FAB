import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1
# Output: unsigned int[256] histogram -> treated as ap_uint<32>[256]

# XF_8UC1 -> ap_uint<8>, dtype: uint8 (input image)
# unsigned int -> ap_uint<32>, dtype: uint32 (histogram output, 256 bins)
function_signature = (
    "void histogram_accel("
    "int32_t* img_in, "
    "int32_t* histogram)"
)

description = f"""
This HLS kernel computes the intensity histogram of a grayscale image.

Parameters:
    img_in:    Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
               a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    histogram: Pointer to a flat array of 256 uint32 values for the output histogram.
               histogram[v] is the count of pixels with intensity value v.

The kernel counts the number of occurrences of each intensity value in the input image.
Each bin in the output corresponds to one intensity level (0-255). The bin values are
unsigned 32-bit integers. The sum of all bins equals {HEIGHT * WIDTH} (total pixels).
No normalization is applied; bins contain raw pixel counts.
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
        "name": "histogram",
        "array_dimension": 1,
        "shape": (256,),
        "total_elements": 256,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((128, 128))
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist.flatten().astype(np.uint32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
