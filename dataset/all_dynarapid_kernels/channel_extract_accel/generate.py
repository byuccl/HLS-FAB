import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC3 (RGB, 3 channels), OUT_TYPE = XF_8UC1
# RGB = 1 (channel ordering: R=0, G=1, B=2)
# channel is a runtime uint16_t parameter; hardcoded here as channel 0 (Red)
CHANNEL = 0

# XF_8UC3 -> ap_uint<8>, dtype: uint8, shape: (H, W, 3)
# XF_8UC1 -> ap_uint<8>, dtype: uint8, shape: (H, W)
function_signature = (
    "void channel_extract_accel("
    "int32_t* img_rgba, "
    "int32_t* img_gray)"
)

description = f"""
This HLS kernel extracts a single channel from a 3-channel RGB image.

Parameters:
    img_rgba: Pointer to a flat array of {HEIGHT * WIDTH * 3} uint8 pixel values
              representing a {HEIGHT}x{WIDTH} 3-channel (RGB) input image in row-major
              order. Each pixel occupies 3 consecutive bytes: [R, G, B].
    img_gray: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
              the extracted single-channel output image ({HEIGHT}x{WIDTH}, row-major).

The kernel extracts channel CHANNEL={CHANNEL} (0=R, 1=G, 2=B) from the packed 3-channel
pixel. For each input pixel, the selected channel byte is written directly to the output.
The output values are uint8 in the range [0, 255] with no transformation applied.
"""

inputs_info = [
    {
        "name": "img_rgba",
        "array_dimension": 3,
        "shape": (HEIGHT, WIDTH, 3),
        "total_elements": HEIGHT * WIDTH * 3,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_gray",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_rgba):
    img = img_rgba.reshape((128, 128, 3))
    return img[:, :, 0].flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
