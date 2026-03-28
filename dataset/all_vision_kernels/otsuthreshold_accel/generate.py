import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# Output: unsigned char& Otsuval -> modeled as ap_uint<8>* (1-element scalar)
# INPUT_PTR_WIDTH = 8, XF_USE_URAM = 0

function_signature = (
    "void otsuthreshold_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* Otsuval)"
)

description = f"""\
This HLS kernel computes the Otsu threshold for a {HEIGHT}\u00d7{WIDTH} grayscale image.
The threshold maximizes the between-class variance of the pixel intensity histogram
using a fixed-point approximation.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}\u00d7{WIDTH} grayscale image in row-major order.
    Otsuval: Pointer to a single uint8 value: the Otsu threshold (0..255).

Algorithm:
1. Compute the 256-bin histogram H[i] = count of pixels with value i.
2. Normalize the histogram using fixed-point inverse (Q0.25 format).
3. For each candidate threshold t = 0..255:
   - Compute background weight wB and foreground weight wF (in Q0.25 format)
   - Compute d = |sumB * total - sum * wB| (where sum and sumB are weighted sums)
   - Compute score = d^2 / (wB * wF) using integer division via reciprocal LUT
4. Return t* = argmax(score). On ties, the first (smallest) threshold wins.
   Stop early if wB accounts for the full total.

Output is stored as uint8 (0..255).
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
        "name": "Otsuval",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((128, 128)).astype(np.uint8)
    thresh_val, _ = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.array([np.uint8(thresh_val)], dtype=np.uint8)


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
