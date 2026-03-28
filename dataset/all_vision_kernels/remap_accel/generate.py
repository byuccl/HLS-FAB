import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
PLANES = 3
WIN_ROWS = 8

function_signature = "void remap_accel(ap_uint<24>* img_inp, ap_uint<24>* img_out, ap_uint<32>* map_x, ap_uint<32>* map_y)"

description = f"""Performs image remapping with bilinear interpolation on a {HEIGHT}x{WIDTH} 3-channel (RGB) 8-bit image.
The kernel takes an input image and two coordinate maps (map_x and map_y) that specify, for each output pixel, the corresponding location in the input image.
Map values are 32-bit integers encoding fixed-point coordinates as float_coord * 256 (Q24.8 format).
For each output pixel at (r, c):
1. The source x-coordinate is map_x[r*{WIDTH}+c] / 256.0 and source y-coordinate is map_y[r*{WIDTH}+c] / 256.0.
2. The integer parts (x0, y0) identify the top-left source pixel.
3. The fractional parts (xf, yf) are quantized to 1/32 precision (5-bit fractional).
4. Bilinear interpolation combines the four neighbors: src[y0,x0], src[y0,x0+1], src[y0+1,x0], src[y0+1,x0+1].
5. Weights are: (1-yf)*(1-xf), (1-yf)*xf, yf*(1-xf), yf*xf.
6. The result is truncated to uint8 per channel.
7. If source coordinates are out of range, output pixel is 0.
Input image: flat array of {HEIGHT*WIDTH*PLANES} uint8 values in row-major, channel-interleaved order.
Map arrays: flat arrays of {HEIGHT*WIDTH} int32 values each.
Output image: flat array of {HEIGHT*WIDTH*PLANES} uint8 values in the same order."""

inputs_info = [
    {
        "name": "img_inp",
        "iotype": "in",
        "array_dimension": 3,
        "dtype": "uint8",
        "shape": (HEIGHT, WIDTH, PLANES),
        "total_elements": HEIGHT * WIDTH * PLANES,
        "min": 0,
        "max": 255,
    },
    {
        "name": "map_x",
        "iotype": "in",
        "array_dimension": 2,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": (WIDTH - 1) * 256,
    },
    {
        "name": "map_y",
        "iotype": "in",
        "array_dimension": 2,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": (HEIGHT - 1) * 256,
    },
    {
        "name": "img_out",
        "iotype": "out",
        "array_dimension": 3,
        "dtype": "uint8",
        "shape": (HEIGHT, WIDTH, PLANES),
        "total_elements": HEIGHT * WIDTH * PLANES,
    },
]


def behavioral(img_inp, map_x, map_y):
    src = img_inp.reshape((128, 128, 3)).astype(np.uint8)
    mx = (map_x.reshape((128, 128)).astype(np.float32)) / 256.0
    my = (map_y.reshape((128, 128)).astype(np.float32)) / 256.0
    out = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    return out.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
