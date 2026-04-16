import numpy as np
import struct
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
TRANSFORM_TYPE = 1   # 1 = PERSPECTIVE
INTERPOLATION = 1    # 1 = BILINEAR

# Fixed-point interpolation constants (from xf_warp_transform.hpp)
INTER_BITS = 5
INTER_TAB_SIZE = 1 << INTER_BITS          # 32
INTER_REMAP_COEF_BITS = 15
INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS  # 32768

# Hardcoded perspective transform matrix (2-pixel translation):
# maps output pixel (col, row) to input source (col+2, row+2)
# Row-major 3x3: M[0..8]
# src_x = (M[0]*col + M[1]*row + M[2]) / (M[6]*col + M[7]*row + M[8])
# src_y = (M[3]*col + M[4]*row + M[5]) / (M[6]*col + M[7]*row + M[8])
TRANSFORM_MATRIX = np.array(
    [1.0, 0.0, 2.0,
     0.0, 1.0, 2.0,
     0.0, 0.0, 1.0],
    dtype=np.float32
)

function_signature = (
    "void warp_transform_accel("
    "int32_t* img_in, "
    "int32_t* transform, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel applies a perspective warp transform to a {HEIGHT}x{WIDTH} grayscale image
using bilinear interpolation (INTERPOLATION=1, TRANSFORM_TYPE=1).

Parameters:
    img_in:    Flat array of {HEIGHT * WIDTH} uint8 pixels in row-major order.
    transform: Array of 9 float32 values (row-major 3x3 perspective matrix M).
               Each float is stored as a raw 32-bit word (ap_uint<32>).
    img_out:   Flat array of {HEIGHT * WIDTH} uint8 output pixels in row-major order.

Algorithm (perspective + bilinear, from xf_warp_transform.hpp):
For each output pixel at (row k, col j):
  1. Compute projective source coordinates:
       ox = M[0]*j + M[1]*k + M[2]
       oy = M[3]*j + M[4]*k + M[5]
       oz = M[6]*j + M[7]*k + M[8]
  2. Scale by INTER_TAB_SIZE/oz (for bilinear fixed-point):
       X = round(ox * INTER_TAB_SIZE / oz)   (fixed-point with INTER_BITS=5 fractional bits)
       Y = round(oy * INTER_TAB_SIZE / oz)
  3. Extract integer and fractional parts:
       J = X >> INTER_BITS  (integer source column)
       I = Y >> INTER_BITS  (integer source row)
       b = X & (INTER_TAB_SIZE - 1)   (fractional column, 0..31)
       a = Y & (INTER_TAB_SIZE - 1)   (fractional row, 0..31)
  4. Compute bilinear coefficients (fixed-point, scale=INTER_REMAP_COEF_SCALE=32768):
       taby = (INTER_REMAP_COEF_SCALE / INTER_TAB_SIZE) * a  = 1024 * a
       tabx = (1.0 / INTER_TAB_SIZE) * b                     = b / 32
       A = floor((INTER_REMAP_COEF_SCALE - taby) * (1 - tabx))
       B = floor((INTER_REMAP_COEF_SCALE - taby) * tabx)
       C = floor(taby * (1 - tabx))
       D = floor(taby * tabx)
  5. Bounds check (in-bounds if J>=0, J<WIDTH-1, I>=0, I<HEIGHT-1):
       out = (A*px[I,J] + B*px[I,J+1] + C*px[I+1,J] + D*px[I+1,J+1]
              + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS
       out-of-bounds: output pixel = 0
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
        "name": "transform",
        "array_dimension": 1,
        "shape": (9,),
        "total_elements": 9,
        "dtype": "float32",
        "iotype": "in",
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


def behavioral(img_in, transform_in=None):
    if transform_in is None:
        M = np.array(TRANSFORM_MATRIX, dtype=np.float32).reshape(3, 3)
    else:
        M = np.asarray(transform_in, dtype=np.float32).reshape(3, 3)
    img = img_in.reshape((128, 128)).astype(np.uint8)
    out = cv2.warpPerspective(img, M, (128, 128), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
