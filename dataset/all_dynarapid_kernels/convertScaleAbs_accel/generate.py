import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

# Scalar parameters
SCALE = 1.5
SHIFT = 10.0

function_signature = (
    "void convertScaleAbs_accel("
    "int32_t* img_in, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel scales, shifts, and takes the absolute value of each pixel in a grayscale image.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The kernel computes each output pixel using the formula:
    out[i] = saturate_cast<uint8>(scale * img_in[i] + shift)

Where:
- scale = {SCALE}
- shift = {SHIFT}
- saturate_cast clamps the result to the range [0, 255]

Processing steps for each pixel:
1. Multiply input pixel by scale (fixed-point: scale * 2^23): temp = {SCALE} * pixel
2. Add shift (fixed-point: shift * 2^23): temp = temp + {SHIFT}
3. Right-shift by 23 bits to recover the integer result
4. Saturate to [0, 255] and store as uint8: values below 0 become 0, above 255 become 255

Note: The Vitis implementation does NOT apply absolute value. With scale={SCALE} and
shift={SHIFT}, the result is always positive for uint8 inputs, so no clamping to 0 occurs.
"""

inputs_info = [
    {
        "name": "img_in",
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

def behavioral(img_in):
    img = img_in.reshape((128, 128)).astype(np.float64)
    out = np.clip(SCALE * img + SHIFT, 0, 255).astype(np.uint8)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
