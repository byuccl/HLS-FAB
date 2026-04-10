import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1, T_8U = 1 (unsigned 8-bit)
# INPUT_PTR_WIDTH = 8
# 6 scalar outputs: int32_t min_value, max_value; uint16_t min_loc_x, min_loc_y, max_loc_x, max_loc_y

function_signature = (
    "void min_max_loc_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<32>* min_value, "
    "ap_uint<32>* max_value, "
    "ap_uint<16>* min_loc_x, "
    "ap_uint<16>* min_loc_y, "
    "ap_uint<16>* max_loc_x, "
    "ap_uint<16>* max_loc_y)"
)

description = f"""\
This HLS kernel finds the minimum and maximum pixel values in a {HEIGHT}×{WIDTH}
grayscale image and returns both values and the (column, row) locations of their
first occurrences.

Parameters:
    img_in:    Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
               a {HEIGHT}×{WIDTH} grayscale image in row-major order.
    min_value: Pointer to a single int32 value: the minimum pixel value found.
    max_value: Pointer to a single int32 value: the maximum pixel value found.
    min_loc_x: Pointer to a single uint16: the column (x) of the minimum pixel.
    min_loc_y: Pointer to a single uint16: the row (y) of the minimum pixel.
    max_loc_x: Pointer to a single uint16: the column (x) of the maximum pixel.
    max_loc_y: Pointer to a single uint16: the row (y) of the maximum pixel.

The scan is in row-major order. For minimum: initialized to 127; only strict less-than
(<) updates the running minimum, so any pixel value < 127 can be the result.
For maximum: initialized to -128; any pixel value (0..255 stored as int32) is > -128,
so the maximum starts updating from the first pixel. On ties, the first occurrence
(leftmost in the earliest row) is returned. Both min_value and max_value are stored
as int32 (the uint8 pixel value zero-extended to 32 bits).
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
        "name": "min_value",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "max_value",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "min_loc_x",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint16",
        "iotype": "out",
    },
    {
        "name": "min_loc_y",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint16",
        "iotype": "out",
    },
    {
        "name": "max_loc_x",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint16",
        "iotype": "out",
    },
    {
        "name": "max_loc_y",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint16",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((128, 128))
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(img)
    return (
        np.array([int(min_val)], dtype=np.int32),
        np.array([int(max_val)], dtype=np.int32),
        np.array([min_loc[0]], dtype=np.uint16),
        np.array([min_loc[1]], dtype=np.uint16),
        np.array([max_loc[0]], dtype=np.uint16),
        np.array([max_loc[1]], dtype=np.uint16),
    )

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
