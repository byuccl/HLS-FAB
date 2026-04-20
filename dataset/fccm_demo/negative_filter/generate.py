import numpy as np
from hlsfab import PromptConstructor, generate_examples

function_signature = "void negative_filter(ap_int<32> *in_pixel, ap_int<32> *out_pixel)"

short_description = "This HLS kernel implements a function to perform a negative filter on a pixel of an image."

description = f"""
This HLS kernel implements a function to perform a negative filter on a pixel of an image.
Parameters to the function include two ap_int<32> pointers:
    Each integer contains RGB data for the pixel to have the filter applied. 
    Each of the RGB components are eight bits. 
    The lowest eight bits are the blue component, the next eight bits are the green component, the next eight bits are the red component.
    The upper eight bits don't store any meaningful data.
    The first integer is the input pixel, the second is the output pixel.
For each pixel, the RGB components need to be unpacked, negated, then repacked into the returned pixel.
"""

inputs_info = [
    {
        "name": "in_pixel",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "in",
        "min": -128,
        "max": 127,
    },
    {
        "name": "out_pixel",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "out",
    }
]

def behavioral(in_pixel):
    # Extract components
    z0 = (in_pixel >> 24) & 0xFF
    r1 = (in_pixel >> 16) & 0xFF
    g1 = (in_pixel >> 8)  & 0xFF
    b1 =  in_pixel        & 0xFF

    # Invert each channel
    z0 = 255 - z0
    r1 = 255 - r1
    g1 = 255 - g1
    b1 = 255 - b1

    # Repack into 32‑bit int
    out_pixel = (z0 << 24) | (r1 << 16) | (g1 << 8) | b1
    return out_pixel

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")