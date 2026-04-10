import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# TWO_INPUT = 1, THREE_INPUT = 0, FOUR_INPUT = 0
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC2
# XF_8UC2: 2 channels x 8 bits = 16 bits per pixel

# XF_8UC1 -> ap_uint<8>, dtype: uint8 (for both inputs)
# XF_8UC2 -> ap_uint<16>, dtype: uint16 (output: 2 channels packed per pixel)
function_signature = (
    "void channel_combine_accel("
    "ap_uint<8>* img_in1, "
    "ap_uint<8>* img_in2, "
    "ap_uint<16>* img_out)"
)

description = f"""
This HLS kernel merges two single-channel grayscale images into one two-channel image.

Parameters:
    img_in1: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
             representing the first channel (channel 0) of the {HEIGHT}x{WIDTH} image.
    img_in2: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
             representing the second channel (channel 1) of the {HEIGHT}x{WIDTH} image.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values for
             the merged two-channel output image.

Each output pixel packs two uint8 channel values into one uint16 value:
    out[i] = (uint16)(img_in1[i]) | ((uint16)(img_in2[i]) << 8)

Channel 0 (img_in1) is stored in the lower 8 bits and channel 1 (img_in2) in the upper
8 bits of each uint16 output element. The result is stored as uint16 with no saturation.
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
    ch0 = img_in1.reshape((128, 128)).astype(np.uint16)
    ch1 = img_in2.reshape((128, 128)).astype(np.uint16)
    out = ch0 | (ch1 << 8)
    return out.astype(np.uint16).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
