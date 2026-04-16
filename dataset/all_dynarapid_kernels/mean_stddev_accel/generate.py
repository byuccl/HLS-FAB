import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# Outputs: unsigned short* mean, unsigned short* stddev -> int32_t* each
# INPUT_PTR_WIDTH = 8

function_signature = (
    "void mean_stddev_accel("
    "int32_t* img_in, "
    "int32_t* mean, "
    "int32_t* stddev)"
)

description = f"""\
This HLS kernel computes the mean and standard deviation of a {HEIGHT}×{WIDTH}
grayscale image using fixed-point integer arithmetic that exactly matches the Vitis
xFStddevkernel C++ HLS implementation.

Parameters:
    img_in: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
            a {HEIGHT}×{WIDTH} grayscale image in row-major order.
    mean:   Pointer to a single uint16 value computed as:
              sum      = sum of all pixel values (uint64)
              tempmean = truncate_to_uint16((256 * sum) // ({HEIGHT} * {WIDTH}))
              mean     = tempmean
    stddev: Pointer to a single uint16 value computed as:
              var_acc  = sum of (pixel * pixel) for all pixels, where each product is
                         computed as (unsigned short)(pixel) * (unsigned short)(pixel)
                         and accumulated into a uint64
              temp     = truncate_to_uint32((65536 * var_acc) // ({HEIGHT} * {WIDTH}))
              Varstddev = (temp - tempmean * tempmean) mod 2^32  (uint32 wraparound)
              t1       = (Varstddev >> 16) << 16  (zero the lower 16 bits)
              stddev   = integer_sqrt(t1)  (floor of sqrt, 16-iteration non-restoring)

To recover actual values: pixel_mean ≈ mean / 256,  pixel_stddev ≈ stddev / 256.
Output values are uint16 (range 0..65535); no saturation is applied.
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
        "name": "mean",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "stddev",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "out",
    },
]

def _integer_sqrt(D):
    """16-iteration non-restoring integer square root matching xf::cv::Sqrt(uint32_t D)."""
    Q = 0
    R = 0
    for i in range(15, -1, -1):
        tmp = (D >> (i + i)) & 3
        if R >= 0:
            R = ((R << 2) | tmp) - ((Q << 2) | 1)
        else:
            R = ((R << 2) | tmp) + ((Q << 2) | 3)
        if R >= 0:
            Q = (Q << 1) | 1
        else:
            Q = (Q << 1) | 0
    # Q is a signed 16-bit short in HLS; replicate sign behavior
    Q = Q & 0xFFFF
    if Q >= 0x8000:
        Q = Q - 0x10000
    return Q

def behavioral(img_in):
    pixels = img_in.flatten().astype(np.uint64)
    N = np.uint64(HEIGHT * WIDTH)

    # Accumulate sum and sum-of-squares exactly as the HLS kernel does
    sum_val = np.uint64(np.sum(pixels))
    # Each pixel^2 computed as unsigned short * unsigned short (fits in uint16 then accumulated)
    var_acc = np.uint64(np.sum(pixels.astype(np.uint64) * pixels.astype(np.uint64)))

    # tempmean = truncate to uint16: (256 * sum) // N
    tempmean = np.uint16((np.uint64(256) * sum_val) // N)

    # temp = truncate to uint32: (65536 * var_acc) // N
    temp = np.uint32((np.uint64(65536) * var_acc) // N)

    # Varstddev = (temp - tempmean^2) mod 2^32
    Varstddev = np.uint32(int(temp) - int(tempmean) * int(tempmean))

    # t1: zero lower 16 bits
    t1 = np.uint32((int(Varstddev) >> 16) << 16)

    # stddev = integer sqrt (non-restoring, 16 iterations)
    stddev_val = _integer_sqrt(int(t1))

    out_mean = np.array([np.uint16(tempmean)], dtype=np.uint16)
    out_stddev = np.array([np.uint16(stddev_val)], dtype=np.uint16)
    return out_mean, out_stddev

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
