import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from xf_config_params.h) ---
HEIGHT = 1080
WIDTH = 1920

# Function signature with int32_t pointer parameters for Dynarapid compatibility
function_signature = "void custom_bgr2y8_accel(int32_t* src, int32_t* dst, int32_t* array_params)"

description = f"""
This HLS kernel implements image processing on {HEIGHT}x{WIDTH} images.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image.

The kernel processes the input image and produces an output image of the same dimensions.
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (1080, 1920, 3),
        "total_elements": 6220800,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (1080, 1920),
        "total_elements": 2073600,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((1080, 1920, 3))
    # Placeholder implementation - replace with actual algorithm
    result = img.astype(np.uint8)
    return result.astype(np.uint8).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)
pc.write_json("kernel.json")
