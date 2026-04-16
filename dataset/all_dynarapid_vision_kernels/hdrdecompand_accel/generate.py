import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from xf_config_params.h) ---
HEIGHT = 1080
WIDTH = 1920

function_signature = "void hdrdecompand_accel(int32_t* img_inp, int32_t* img_out)"

description = f"""
This HLS kernel implements image processing on {HEIGHT}x{WIDTH} images.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint16 pixel values representing
             a {HEIGHT}x{WIDTH} image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} int32 pixel values for
             the output image.

The kernel processes the input image and produces an output image of the same dimensions.
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (1080, 1920),
        "total_elements": 2073600,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 65535,
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
    img = img_inp.reshape((1080, 1920))
    result = img.astype(np.int32)
    return result.astype(np.int32).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)
pc.write_json("kernel.json")
