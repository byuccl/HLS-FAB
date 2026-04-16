import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 2160
WIDTH = 3840

HV_DIST = int(0.954999983 * (2**16))    # 62586
DIAG_DIST = int(1.36930001 * (2**16))   # 89744
SET_MAX_VAL = 2147483647 >> 2            # 536870911
SCALE = 1.0 / (1 << 16)

function_signature = "void distancetransform_accel(int32_t* img_inp, float* img_out, int32_t* fw_pass_data)"

description = f"""Computes the distance transform of a binary image using a two-pass algorithm.
The input is a {HEIGHT}x{WIDTH} single-channel 8-bit binary image (0 = background, non-zero = foreground).
The output is a {HEIGHT}x{WIDTH} array of float values representing the Euclidean-approximated distance of each foreground pixel to the nearest background pixel.
Background pixels (value 0) have output distance 0.0.
The algorithm uses two passes:
1. Forward pass (top-left to bottom-right): For each non-zero pixel, compute the minimum distance from four neighbors (top-left + {DIAG_DIST}, top + {HV_DIST}, top-right + {DIAG_DIST}, left + {HV_DIST}), all in 32-bit fixed-point (Q16.16). Border pixels use a large sentinel value ({SET_MAX_VAL}).
2. Backward pass (bottom-right to top-left): For each pixel whose forward-pass value exceeds {HV_DIST}, refine by taking the minimum of the forward value and four backward neighbors (bottom-right + {DIAG_DIST}, bottom + {HV_DIST}, bottom-left + {DIAG_DIST}, right + {HV_DIST}).
The final 32-bit fixed-point distance is converted to float by multiplying by {SCALE} (i.e., dividing by 65536).
The distance weights approximate the Euclidean metric: horizontal/vertical step cost is approximately 0.955 and diagonal step cost is approximately 1.369.
Input: flat array of {HEIGHT * WIDTH} uint8 values.
Output: flat array of {HEIGHT * WIDTH} float32 values."""

inputs_info = [
    {
        "name": "img_inp",
        "iotype": "in",
        "array_dimension": 2,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "iotype": "out",
        "array_dimension": 2,
        "dtype": "float32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
    },
    {
        "name": "fw_pass_data",
        "iotype": "out",
        "array_dimension": 2,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
    },
]


def behavioral(img_inp):
    img = img_inp.reshape((2160, 3840)).astype(np.uint8)
    dist = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    fw_data = np.zeros(2160 * 3840, dtype=np.uint32)
    return dist.astype(np.float32).flatten(), fw_data


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
