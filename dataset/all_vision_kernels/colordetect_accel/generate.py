import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_SIZE = 3
MAXCOLORS = 3
# IN_TYPE = XF_8UC3 (3-channel BGR), OUT_TYPE = XF_8UC1
# INPUT_PTR_WIDTH = 32, OUTPUT_PTR_WIDTH = 8
# Pipeline: BGR -> bgr2hsv -> colorthresholding -> erode -> dilate -> dilate -> erode

# Threshold arrays from testbench (hardcoded per xf_colordetect_tb.cpp)
# Each group of 3 values: [H_low, S_low, V_low], [H_high, S_high, V_high]
LOW_THRESH = np.array([
    22, 150, 60,   # Yellow lower
    38, 150, 60,   # Green lower
    160, 150, 60,  # Red lower
], dtype=np.uint8)

HIGH_THRESH = np.array([
    38, 255, 255,  # Yellow upper
    75, 255, 255,  # Green upper
    179, 255, 255, # Red upper
], dtype=np.uint8)

# XF_8UC3 input -> ap_uint<32> (INPUT_PTR_WIDTH=32), 3-channel BGR flat array
function_signature = (
    "void colordetect_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel performs color detection on a {HEIGHT}x{WIDTH} BGR image and produces
a binary mask highlighting pixels that fall within any of {MAXCOLORS} predefined HSV color ranges.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH * 3} uint8 values representing a
             {HEIGHT}x{WIDTH} BGR image in row-major order (3 bytes per pixel: B, G, R).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values for the output binary
             mask (0 = no color match, 255 = color match).

Processing pipeline:
  1. BGR to HSV conversion using integer lookup tables (exact Xilinx xf::cv::bgr2hsv formula).
  2. Color thresholding: for each pixel, check if HSV values fall within any of {MAXCOLORS} color
     ranges (Yellow: H in [22,38], Green: H in [38,75], Red: H in [160,179]; all with S in
     [150,255], V in [60,255]). Output is 255 if any range matches, 0 otherwise.
  3. Morphological post-processing: erode -> dilate -> dilate -> erode with a {FILTER_SIZE}x{FILTER_SIZE}
     all-ones rectangular structuring element (XF_BORDER_CONSTANT).
     - Erosion border constant = 255 (does not affect minimum).
     - Dilation border constant = 0 (does not affect maximum).

Output values are uint8 in {{0, 255}}.
"""

inputs_info = [
    {
        "name": "img_in",
        "array_dimension": 3,
        "shape": (HEIGHT, WIDTH, 3),
        "total_elements": HEIGHT * WIDTH * 3,
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
    bgr = img_in.reshape((128, 128, 3)).astype(np.uint8)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # Three color ranges
    ranges = [
        ((22, 150, 60), (38, 255, 255)),
        ((38, 150, 60), (75, 255, 255)),
        ((160, 150, 60), (179, 255, 255)),
    ]
    mask = np.zeros((128, 128), dtype=np.uint8)
    for lo, hi in ranges:
        mask |= cv2.inRange(hsv, np.array(lo, dtype=np.uint8), np.array(hi, dtype=np.uint8))
    # Morphology: erode -> dilate -> dilate -> erode with cross SE
    se = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    mask = cv2.erode(mask, se, borderType=cv2.BORDER_CONSTANT, borderValue=255)
    mask = cv2.dilate(mask, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.dilate(mask, se, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    mask = cv2.erode(mask, se, borderType=cv2.BORDER_CONSTANT, borderValue=255)
    return mask.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
