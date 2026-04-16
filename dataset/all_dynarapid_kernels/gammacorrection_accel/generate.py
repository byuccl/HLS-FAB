import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h and testbench) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC3 -> ap_uint<8>, dtype: uint8, 3 channels
# OUT_TYPE = XF_8UC3 -> ap_uint<8>, dtype: uint8, 3 channels
# INPUT_PTR_WIDTH = 32, OUTPUT_PTR_WIDTH = 32
# gamma_lut[256*3] is a runtime parameter; hardcoded below with testbench values:
GAMMA_VAL_R = 0.5   # applied to channel 0
GAMMA_VAL_G = 0.8   # applied to channel 1
GAMMA_VAL_B = 0.8   # applied to channel 2

# XF_8UC3 -> int32_t* for each channel byte
function_signature = (
    "void gammacorrection_accel("
    "int32_t* img_inp, "
    "int32_t* img_out)"
)

description = f"""\
This HLS kernel applies per-channel gamma correction to a 3-channel uint8 image
using a precomputed 256-entry lookup table (LUT) for each channel.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH * 3} uint8 bytes representing
             a {HEIGHT}×{WIDTH} 3-channel image in row-major, interleaved channel order
             (channel 0, channel 1, channel 2 per pixel).
    img_out: Pointer to a flat array of {HEIGHT * WIDTH * 3} uint8 bytes for the
             gamma-corrected output image (same layout as input).

The gamma LUT for each channel is computed as follows (hardcoded gamma values):
    gamma_val_ch0 = {GAMMA_VAL_R},  gamma_val_ch1 = {GAMMA_VAL_G},  gamma_val_ch2 = {GAMMA_VAL_B}

For channel k with gamma_val_k:
    inv_gamma = 1.0 / gamma_val_k  (float32)
    pow_val = float32(i / 255.0) ** inv_gamma  (float32 exponentiation)
    lut[k][i] = clip(truncate(pow_val * 255.0), 0, 255)  (truncation to short, then uint8 clamp)

Each output pixel channel value = lut[channel][input_channel_value].
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 3,
        "shape": (HEIGHT, WIDTH, 3),
        "total_elements": HEIGHT * WIDTH * 3,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 3,
        "shape": (HEIGHT, WIDTH, 3),
        "total_elements": HEIGHT * WIDTH * 3,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((128, 128, 3)).astype(np.uint8)
    out = np.empty_like(img)
    for ch, gamma in enumerate([0.5, 0.8, 0.8]):
        lut = np.clip(np.power(np.arange(256) / 255.0, 1.0 / gamma) * 255, 0, 255).astype(np.uint8)
        out[:, :, ch] = cv2.LUT(img[:, :, ch], lut)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
