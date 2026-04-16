import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# The Vitis equalizeHist kernel takes the input image twice (two passes):
#   img_inp  -> first pass (used to build the histogram / CDF)
#   img_inp1 -> second pass (pixels to remap via the equalization LUT)
# Both buffers contain the same image data in practice.

# XF_8UC1 -> ap_uint<8>, dtype: uint8
function_signature = (
    "void equalizeHist_accel("
    "int32_t* img_inp, "
    "int32_t* img_inp1, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel applies histogram equalization to a grayscale image using fixed-point
arithmetic that exactly matches the Vitis xFEqualize C++ HLS implementation.

Parameters:
    img_inp:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
              the input image ({HEIGHT}x{WIDTH}, row-major). Used to compute the histogram.
    img_inp1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values containing
              the same input image (second pass for pixel remapping).
    img_out:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
              the equalized output image (same dimensions as input).

The kernel performs histogram equalization using the Vitis fixed-point algorithm:

1. First pass (img_inp): compute a 256-bin histogram H[v] = count of pixels with value v.
2. Compute:
     temp_val = {HEIGHT} * {WIDTH} = {HEIGHT * WIDTH}
     init_val = temp_val - H[0]   (total non-zero-bin pixels)
     scale    = (1 << 31) // init_val  if init_val > 0, else 0  (integer division)
     scale1   = 255 * scale  (40-bit intermediate)
3. Build cum_hist[256] LUT (uint8):
     cum_hist[0] = 0
     For i = 1..255:
       temp_sum += H[i]   (running cumulative sum, excludes H[0])
       sum = temp_sum * scale1  (64-bit multiply)
       sum = sum + 0x40000000   (rounding offset = 2^30)
       cum_hist[i] = sum >> 31  (right-shift by 31, result saturated to uint8)
4. Second pass (img_inp1): output[pixel] = cum_hist[pixel].

Output values are uint8 in [0, 255].
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
        "name": "img_inp1",
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
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_inp, img_inp1):
    # Compute 256-bin histogram from img_inp
    hist = np.zeros(256, dtype=np.uint32)
    for v in img_inp.flatten():
        hist[v] += 1

    # Vitis fixed-point equalization (xFEqualize)
    temp_val = np.uint32(HEIGHT * WIDTH)
    init_val = np.uint32(temp_val - hist[0])
    if init_val == 0:
        scale = np.uint32(0)
    else:
        scale = np.uint32((1 << 31) // int(init_val))

    # scale1 = 255 * scale, treated as a 40-bit value
    scale1 = int(255) * int(scale)

    cum_hist = np.zeros(256, dtype=np.uint8)
    # cum_hist[0] = 0 (already set)
    temp_sum = np.uint32(0)
    for i in range(1, 256):
        temp_sum = np.uint32(int(temp_sum) + int(hist[i]))
        # 64-bit multiply
        s = int(temp_sum) * scale1
        # add rounding offset
        s = s + 0x40000000
        # right-shift by 31, result is uint8 (saturates at 255 naturally)
        val = s >> 31
        cum_hist[i] = np.uint8(val if val <= 255 else 255)

    # Apply LUT to img_inp1
    out = cum_hist[img_inp1.flatten()]
    return out.astype(np.uint8)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
