import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128

# Scalar parameters (hardcoded from testbench):
# BlackLevel = 32, MaxLevel = 255
# MulValue = 255.0 / (255 - 32) = 1.143497...
# mul_value (blc_config_1) = int(MulValue * 65536) = 74936
BLACK_LEVEL = 32
MUL_VALUE = 74936

function_signature = (
    "void blackLevelCorrection_accel("
    "int32_t* in_img_ptr, "
    "int32_t* out_img_ptr)"
)

description = f"""Black level correction on a {HEIGHT}x{WIDTH} single-channel 8-bit grayscale image.

Parameters:
    in_img_ptr:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values
                 representing a {HEIGHT}x{WIDTH} grayscale input image (row-major).
    out_img_ptr: Pointer to a flat array of {HEIGHT * WIDTH} uint8 output pixel values.

For each pixel, the black level ({BLACK_LEVEL}) is subtracted from the input value,
and the result is scaled by a fixed-point multiplier to remap the range
[{BLACK_LEVEL}, 255] to approximately [0, 255].

The multiplier is derived as: mul_value = int(255.0 / (255 - {BLACK_LEVEL}) * 65536) = {MUL_VALUE}.
This integer is converted to a Q1.15 unsigned fixed-point value (ap_ufixed<16,1>) by
dividing by 65536, yielding approximately {MUL_VALUE / 65536.0:.6f}.

The fixed-point multiplication mulval is computed as:
    mulval_q15 = floor((mul_value / 65536.0) * 32768) (truncate to 15 fractional bits)
    mulval = mulval_q15 / 32768.0

For each pixel:
    If pixel < {BLACK_LEVEL}: output = 0
    Else: output = clamp(floor((pixel - {BLACK_LEVEL}) * mulval), 0, 255)
"""

inputs_info = [
    {
        "name": "in_img_ptr",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "out_img_ptr",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]


def behavioral(in_img_ptr):
    img = in_img_ptr.astype(np.int32)

    # Replicate fixed-point math from the HLS kernel:
    # mul_fact = ap_ufixed<32,18>((float)MUL_VALUE) -> value is MUL_VALUE exactly (fits in 18 int bits)
    # mulval = ap_ufixed<16,1>(mul_fact >> 16)
    # mul_fact >> 16 means divide by 65536
    # ap_ufixed<16,1> means 1 integer bit, 15 fractional bits, truncation
    mul_float = MUL_VALUE / 65536.0
    mulval_q15 = int(mul_float * 32768)  # truncate to Q1.15
    # mulval as a real number
    mulval = mulval_q15 / 32768.0

    # For each pixel:
    # med_val = pixel - BLACK_LEVEL (signed int)
    # if pixel < BLACK_LEVEL: value = 0
    # else: value = int(med_val * mulval) -- truncation toward zero
    # output = clamp(value, 0, 255)
    med = img - BLACK_LEVEL
    value = np.where(img < BLACK_LEVEL, 0, (med * mulval_q15) >> 15)
    out = np.clip(value, 0, 255)

    return out.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
