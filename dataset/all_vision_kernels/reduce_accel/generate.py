import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# OUT_TYPE = XF_32SC1 -> ap_uint<32>, dtype: int32
# INPUT_PTR_WIDTH = 8, OUTPUT_PTR_WIDTH = 32
# DIM = 0 (reduce along rows -> column-wise, output shape 1 x WIDTH)
# REDUCTION_OP = XF_REDUCE_AVG

function_signature = (
    "void reduce_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<32>* img_out)"
)

description = f"""\
This HLS kernel reduces a grayscale image along the row dimension (DIM=0) using
average reduction (REDUCE_AVG), producing a 1×{WIDTH} row of column averages.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}×{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {WIDTH} int32 values representing the column
             averages of the input image.

Operation: For each column c in [0, {WIDTH}), the output is:
    img_out[c] = floor(sum(img_in[r, c] for r in 0..{HEIGHT}-1) / {HEIGHT})

The accumulation is done in int32 and the final division by {HEIGHT} uses integer
division (floor), truncating toward zero. The output is a single row of {WIDTH}
int32 values.
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
        "shape": (1, WIDTH),
        "total_elements": WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((128, 128))
    result = cv2.reduce(img, 0, cv2.REDUCE_AVG, dtype=cv2.CV_32S)
    return result.flatten().astype(np.int32)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
