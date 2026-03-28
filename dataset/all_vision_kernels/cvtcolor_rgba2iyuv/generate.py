import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 1080
WIDTH = 1920

function_signature = "void cvtcolor_rgba2iyuv(ap_uint<32>* imgInput, ap_uint<8>* imgOutput0, ap_uint<8>* imgOutput1, ap_uint<8>* imgOutput2)"

description = f"""Converts an RGBA image of size {HEIGHT}x{WIDTH} to IYUV (YUV 4:2:0 planar) format.

The input is a flat array of {HEIGHT*WIDTH} 32-bit packed RGBA pixels, where each pixel packs
R in bits [7:0], G in bits [15:8], B in bits [23:16], and A in bits [31:24].

The function produces three separate output planes:
- imgOutput0: Y (luma) plane of size {HEIGHT}x{WIDTH} (one uint8 per pixel).
- imgOutput1: U (chroma-blue) plane of size {HEIGHT//4}x{WIDTH} ({HEIGHT*WIDTH//4} uint8 values).
- imgOutput2: V (chroma-red) plane of size {HEIGHT//4}x{WIDTH} ({HEIGHT*WIDTH//4} uint8 values).

Conversion uses fixed-point Q1.15 arithmetic with the following coefficients:
  Y = ((8422*R + 16516*G + 3212*B + 16384) >> 15) + 16
  U = ((-4850*R + -9536*G + 14386*B + 16384) >> 15) + 128
  V = ((14382*R + -12059*G + -2327*B + 16384) >> 15) + 128
Results are saturated to [0, 255].

U and V are subsampled 2x2: each value is computed from the top-left pixel of each 2x2 block.
The U/V planes are stored row-major with dimensions ({HEIGHT//4}, {WIDTH}), meaning that for every
two rows of input, one row of U/V values is produced, and within that row, every other pixel
(even columns) contributes a U/V sample."""

inputs_info = [
    {
        "name": "imgInput",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint32",
        "iotype": "in",
        "min": 0,
        "max": int(np.iinfo(np.uint32).max),
    },
    {
        "name": "imgOutput0",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
    {
        "name": "imgOutput1",
        "array_dimension": 1,
        "shape": ((HEIGHT // 4) * WIDTH,),
        "total_elements": (HEIGHT // 4) * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
    {
        "name": "imgOutput2",
        "array_dimension": 1,
        "shape": ((HEIGHT // 4) * WIDTH,),
        "total_elements": (HEIGHT // 4) * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]


def behavioral(imgInput):
    rgba = imgInput.astype(np.uint32)
    R = (rgba & 0xFF).astype(np.uint8)
    G = ((rgba >> 8) & 0xFF).astype(np.uint8)
    B = ((rgba >> 16) & 0xFF).astype(np.uint8)
    A = ((rgba >> 24) & 0xFF).astype(np.uint8)
    img_rgba = np.stack([R.reshape(1080, 1920), G.reshape(1080, 1920),
                         B.reshape(1080, 1920), A.reshape(1080, 1920)], axis=-1)
    yuv = cv2.cvtColor(img_rgba, cv2.COLOR_RGBA2YUV_I420)
    # I420 layout: Y plane (H rows), U plane (H/4 rows, W cols), V plane (H/4 rows, W cols)
    H, W = 1080, 1920
    Y = yuv[:H, :].flatten()
    U = yuv[H:H + H // 4, :].flatten()
    V = yuv[H + H // 4:, :].flatten()
    return (Y, U, V)


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
