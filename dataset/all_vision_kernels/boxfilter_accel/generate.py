import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

function_signature = "void boxfilter_accel(ap_uint<8>* img_inp, ap_uint<8>* img_out)"

description = f"""
This HLS kernel applies a 3x3 box filter (mean filter) to a grayscale image.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The box filter computes the output pixel at position (i, j) as the average of all pixels
in a {FILTER_WIDTH}x{FILTER_WIDTH} window centered at (i, j):

    out[i, j] = (sum of all 9 pixels in window) / 9

Border handling: Pixels outside the image boundary are treated as zero (zero-padding).
Border pixels are normalized by dividing by the full filter area (9), NOT by the count
of in-bounds pixels. For example, a pixel at the top-left corner (0, 0) has only 4
valid neighbors; the other 5 window positions are zero. The sum of these 4+1=5 pixels
is still divided by 9.

Division and output: The sum is multiplied by 3641 (which is 1/9 in Q1.15 fixed-point
format) and right-shifted by 15 bits, which is equivalent to integer division by 9 with
truncation (floor). The result is then stored as uint8. Values are truncated (not rounded)
and implicitly saturated to [0, 255] by the uint8 cast.
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
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    """
    Pure numpy implementation of the 3x3 box filter.
    
    C++ implementation notes (from xf_box_filter.hpp, xFApplyMask3x3):
    - Division method: sum * 3641 >> 15, which equals floor(sum / 9)
    - Border normalization: always divide by 9 (full window), not by in-bounds pixel count
    - Accumulator type: sum is computed as ap_int<20>, then multiplied and shifted
    - Output saturation: truncated to uint8 (floor cast), auto-saturates on overflow
    """
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.int32)
    
    # Zero-pad the image: 1 pixel on each side for a 3x3 filter
    pad = FILTER_WIDTH // 2
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    
    out = np.zeros((HEIGHT, WIDTH), dtype=np.int32)
    
    # Apply the box filter: sum of 3x3 window divided by 9
    for i in range(HEIGHT):
        for j in range(WIDTH):
            window = padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH]
            window_sum = window.sum()
            # Mimic the C++ fixed-point arithmetic: multiply by 3641, then right-shift by 15
            # This is equivalent to floor(sum / 9)
            result = (window_sum * 3641) >> 15
            out[i, j] = result
    
    # Clip to [0, 255] and cast to uint8 (mimics hardware saturation)
    out = np.clip(out, 0, 255).astype(np.uint8)
    
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
