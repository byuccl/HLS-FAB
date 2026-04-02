# generate.py for: boxfilter_accel
#
# Dynarapid Vision kernel: /home/hoodoo17/Vitis_Libraries/vision/L1/examples/boxfilter/
#
# Source files read to produce this:
#   - xf_box_filter_accel.cpp          -> top function name: boxfilter_accel
#   - config/xf_config_params.h        -> HEIGHT=128, WIDTH=128, IN_TYPE=XF_8UC1,
#                                         OUT_TYPE=XF_8UC1, INPUT_PTR_WIDTH=8, 
#                                         OUTPUT_PTR_WIDTH=8, FILTER_WIDTH=3
#   - description.json                 -> topfunction: boxfilter_accel
#
# Key decisions:
#   - XF_8UC1 -> numpy dtype: uint8 (1 channel, 8 bits per pixel)
#   - C type: uint8_t (NOT ap_uint - Dynarapid does not support arbitrary precision)
#   - img_inp / img_out are both 128x128 uint8 images
#   - FILTER_WIDTH=3 is a compile-time constant -> hardcoded
#   - Border handling: zero-padding (XF_BORDER_CONSTANT)
#   - Division method: fixed-point (multiply by 3641, shift right 15) -> equivalent to / 9
#   - Output: truncated to uint8 (floor cast)

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

function_signature = "void boxfilter_accel(int32_t* img_inp, int32_t* img_out)"

description = f"""
This HLS kernel applies a 3x3 box filter (mean filter) to a grayscale image using zero-padding for border handling.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the filtered output image (same dimensions as input).

Algorithm Details:
    The kernel computes the mean of each 3x3 neighborhood using fixed-point arithmetic:
    - Each 3x3 window is summed (accumulator can hold values 0-2304 for uint8 input)
    - The sum is multiplied by 3641 (which is 1/9 in Q1.15 fixed-point format)
    - The result is right-shifted by 15 bits, effectively computing (sum * 3641) >> 15 = sum / 9
    - The result is truncated (floor) to uint8 for the output pixel
    
    Border Handling: Zero-padding
    - Pixels outside the image are treated as 0
    - All pixels (including border pixels) use the full 3x3 window, padded with zeros
    - The divisor is always 9, regardless of pixel location
    - Border pixels will have lower values due to the zero-padding
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
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

# Division constant from xf_box_filter.hpp
# XF_DIV_VAL_3x3 = 3641 (which is 1/9 in Q1.15 fixed-point format)
# The computation (sum * 3641) >> 15 is equivalent to sum / 9
XF_DIV_VAL_3x3 = 3641

def behavioral(img_inp):
    """
    Behavioral model of boxfilter_accel kernel.
    
    Applies a 3x3 box (mean) filter using zero-padding for borders.
    Uses fixed-point arithmetic: (sum * 3641) >> 15 = sum / 9
    
    Division Method: The sum of 9 pixels is multiplied by 3641 (1/9 in Q1.15)
                    and right-shifted by 15 bits, which truncates to uint8.
    Border Normalization: All pixels use divisor 9 (including border pixels with zero-padding).
    Rounding: Truncated (floor) division via right shift.
    """
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.uint32)
    
    # Pad the image with zeros on all sides for the 3x3 filter
    pad = FILTER_WIDTH // 2  # pad = 1 for 3x3 filter
    padded = np.pad(img, pad, mode='constant', constant_values=0)
    
    # Output array
    out = np.zeros((HEIGHT, WIDTH), dtype=np.uint32)
    
    # Apply 3x3 box filter with fixed-point arithmetic
    for i in range(HEIGHT):
        for j in range(WIDTH):
            # Extract 3x3 window (already padded, so no boundary checks needed)
            window = padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH]
            
            # Sum all 9 pixels
            pixel_sum = np.sum(window)
            
            # Apply fixed-point multiplication and right shift: (sum * 3641) >> 15
            # This is equivalent to sum / 9
            res_fixed = (pixel_sum * XF_DIV_VAL_3x3) >> 15
            
            # Truncate to uint8
            out[i, j] = res_fixed
    
    # Cast to uint8 and flatten
    return out.astype(np.uint8).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)
pc.write_json("kernel.json")
