import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# Outputs: unsigned short* mean, unsigned short* stddev -> ap_uint<16>* each
# INPUT_PTR_WIDTH = 8

function_signature = (
    "void mean_stddev_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<16>* mean, "
    "ap_uint<16>* stddev)"
)

description = f"""\
This HLS kernel computes the mean and standard deviation of a {HEIGHT}×{WIDTH}
grayscale image using fixed-point arithmetic and stores them as scaled uint16 values.

Parameters:
    img_in: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
            a {HEIGHT}×{WIDTH} grayscale image in row-major order.
    mean:   Pointer to a single uint16 value = floor(256 × pixel_mean), where
            pixel_mean = (sum of all pixels) / ({HEIGHT} × {WIDTH}).
    stddev: Pointer to a single uint16 value = floor(sqrt(t1)), where:
              sum_sq   = sum of (pixel²) over all pixels
              tempmean = floor(256 × sum / ({HEIGHT} × {WIDTH}))   (same as mean output)
              temp     = floor(65536 × sum_sq / ({HEIGHT} × {WIDTH})) as uint32
              Varstddev = temp − tempmean²   (uint32 arithmetic, wraps if negative)
              t1       = (Varstddev >> 16) << 16   (zero lower 16 bits)
              stddev   = floor(sqrt(t1)) using 16-iteration non-restoring integer sqrt

To recover actual values: pixel_mean ≈ mean / 256,  pixel_stddev ≈ stddev / 256.
Output values are uint16 (range 0..65535); no saturation is applied.
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
        "name": "mean",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint16",
        "iotype": "out",
    },
    {
        "name": "stddev",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint16",
        "iotype": "out",
    },
]

def behavioral(img_in):
    img = img_in.reshape((128, 128))
    mean_val, stddev_val = cv2.meanStdDev(img)
    # Vitis outputs: mean = floor(256 * pixel_mean)
    out_mean = np.uint16(int(mean_val[0][0] * 256))
    # Vitis stddev uses fixed-point sqrt, approximate with round
    out_stddev = np.uint16(int(round(stddev_val[0][0] * 256)))
    return np.array([out_mean], dtype=np.uint16), np.array([out_stddev], dtype=np.uint16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
