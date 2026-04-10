import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h and testbench) ---
HEIGHT = 128
WIDTH = 128
# BFORMAT = XF_BAYER_RG
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# INPUT_PTR_WIDTH = 256 (bus width only, pixel is still 8-bit)
# Gains from testbench: rgain=154, bgain=140, ggain=200
RGAIN = 154
BGAIN = 140
GGAIN = 200

function_signature = (
    "void gaincontrol_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""\
This HLS kernel applies per-channel Bayer gain control to a {HEIGHT}×{WIDTH} raw
image using the RG Bayer pattern.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}×{WIDTH} raw Bayer-pattern image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 output pixel values.

Bayer pattern (RG): The pixel at (row, col) belongs to:
    R channel  if row is even and col is even
    B channel  if row is odd  and col is odd
    G channel  otherwise

Gain values (hardcoded): R gain = {RGAIN}, G gain = {GGAIN}, B gain = {BGAIN}.

Operation for each pixel:
    gain = RGAIN if (row%2==0 and col%2==0)
    gain = BGAIN if (row%2==1 and col%2==1)
    gain = GGAIN otherwise
    result = (pixel_value * gain) >> 7
    img_out[row, col] = clamp(result, 0, 255)  (saturated to uint8 range)
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
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]

# XF_BAYER_RG pattern:
#   even row, even col -> R channel -> RGAIN
#   odd  row, odd  col -> B channel -> BGAIN
#   else               -> G channel -> GGAIN
# Formula: result = (pixel * gain) >> 7, saturate to [0, 255]
def behavioral(img_in):
    img = img_in.reshape((HEIGHT, WIDTH)).astype(np.int32)

    rows = np.arange(HEIGHT, dtype=np.int32).reshape(-1, 1)
    cols = np.arange(WIDTH, dtype=np.int32).reshape(1, -1)

    gain_map = np.where(
        (rows % 2 == 0) & (cols % 2 == 0), RGAIN,
        np.where((rows % 2 == 1) & (cols % 2 == 1), BGAIN, GGAIN)
    ).astype(np.int32)

    result = (img * gain_map) >> 7
    result = np.clip(result, 0, 255)
    return result.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
