import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 720
WIDTH = 1280
TOTAL_ELEMENTS = HEIGHT * WIDTH  # 921600

function_signature = "void axiconv_accel(ap_uint<8>* src, ap_uint<8>* dst)"

description = f"""Performs a passthrough (identity) operation on an 8-bit grayscale image.

The function reads an input image of {HEIGHT}x{WIDTH} unsigned 8-bit pixels ({TOTAL_ELEMENTS} total elements)
and copies it directly to the output without any modification.

This kernel converts the input data into an internal matrix representation, then converts
it back to the output format, effectively serving as a data passthrough or format conversion
test kernel.

Inputs:
- src: flat array of {TOTAL_ELEMENTS} unsigned 8-bit pixel values (grayscale image {HEIGHT}x{WIDTH})

Outputs:
- dst: flat array of {TOTAL_ELEMENTS} unsigned 8-bit pixel values, identical to src
"""

inputs_info = [
    {
        "name": "src",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS,),
        "total_elements": TOTAL_ELEMENTS,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "dst",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS,),
        "total_elements": TOTAL_ELEMENTS,
        "dtype": "uint8",
        "iotype": "out",
    },
]


def behavioral(src):
    # Identity / passthrough: output is same as input
    return src.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
