import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_16SC1, OUT_TYPE = XF_16SC1
# DEG_TYPE = XF_DEGREES (output in degrees, not radians)

# XF_16SC1 -> ap_uint<16>, dtype: int16
# Two input buffers (x-gradient, y-gradient), one output (angle in degrees)
function_signature = (
    "void phase_accel("
    "ap_uint<16>* img_inp1, "
    "ap_uint<16>* img_inp2, "
    "ap_uint<16>* img_out)"
)

description = f"""
This HLS kernel computes the per-pixel phase angle (in degrees) of 2D gradient vectors.

Parameters:
    img_inp1: Pointer to a flat array of {HEIGHT * WIDTH} int16 values representing
              the x-component (horizontal gradient) of a {HEIGHT}x{WIDTH} image.
    img_inp2: Pointer to a flat array of {HEIGHT * WIDTH} int16 values representing
              the y-component (vertical gradient) of the same {HEIGHT}x{WIDTH} image.
    img_out:  Pointer to a flat array of {HEIGHT * WIDTH} int16 values containing
              the angle of each gradient vector in degrees.

For each pixel, the output angle is computed as:
    angle_deg = atan2(img_inp2[i], img_inp1[i]) * (180 / pi)
    angle_deg = angle_deg + 360 if angle_deg < 0 else angle_deg   (maps to [0, 360))
    output[i] = round(angle_deg * 64)  (stored as int16)

The output is degrees scaled by 64, in the range [0, 23040] (= [0, 360)*64). The kernel uses
a fixed-point lookup table approximation of atan2. The scale factor 64 = 2^6 comes from the
fixed-point normalization: XF_NORM_FACTOR=58671=(180/pi)*2^10, angle in Q4.12 radians,
result = (58671 * angle_Q4.12 + 0x8000) >> 16 = round(degrees * 64). Output stored as int16.
"""

inputs_info = [
    {
        "name": "img_inp1",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int16",
        "iotype": "in",
        "min": -256,
        "max": 255,
    },
    {
        "name": "img_inp2",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int16",
        "iotype": "in",
        "min": -256,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int16",
        "iotype": "out",
    },
]

def behavioral(img_inp1, img_inp2):
    gx = img_inp1.reshape((128, 128)).astype(np.float32)
    gy = img_inp2.reshape((128, 128)).astype(np.float32)
    angle = cv2.phase(gx, gy, angleInDegrees=True)
    return (angle * 64).astype(np.int16).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
