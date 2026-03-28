import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h and xf_harris_tb.cpp) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3   # XF_FILTER_3X3
BLOCK_WIDTH = 3    # XF_FILTER_3X3
NMS_RADIUS = 1

# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1
# INPUT_PTR_WIDTH = 8, OUTPUT_PTR_WIDTH = 8

# Harris parameters (from testbench when FILTER_WIDTH==3)
THRESHOLD = 442    # uint16_t threshold
K = 0.04           # Harris k constant (reference)
K_FIXED = 2621     # int(K * (1 << 16)) = int(0.04 * 65536) = 2621

# Sobel scale for FILTER_3X3 = 6 (from xf_harris.hpp line 69)
SOBEL_SCALE = 6

# Box filter normalization: 3641 * sum >> 15  (1/9 in Q1.15, from xf_box_filter.hpp)
XF_DIV_VAL_3x3 = 3641

function_signature = (
    "void cornerHarris_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = (
    "This HLS kernel detects corners in a grayscale image using the Harris corner\n"
    "detector algorithm.\n\n"
    f"Parameters:\n"
    f"    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing\n"
    f"             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.\n"
    f"    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for\n"
    f"             the corner response map. Corner pixels are marked as 255; all others are 0.\n\n"
    f"The kernel uses the Harris corner detector with the following parameters:\n"
    f"    FILTER_WIDTH = {FILTER_WIDTH} (Sobel kernel size)\n"
    f"    BLOCK_WIDTH  = {BLOCK_WIDTH} (box filter window size)\n"
    f"    NMS_RADIUS   = {NMS_RADIUS} (non-maximum suppression radius)\n"
    f"    threshold    = {THRESHOLD}\n"
    f"    K            = {K} (Harris k factor, fixed-point: k={K_FIXED} = K * 2^16)\n\n"
    "Pipeline (6 stages, all with XF_BORDER_CONSTANT zero-padding):\n\n"
    "Stage 1 - Sobel 3x3 gradients (int16):\n"
    "    gx = (t2 + b2 - t0 - b0) + 2*(m2 - m0)\n"
    "    gy = (b0 + b2 - t0 - t2) + 2*(b1 - t1)\n"
    "    (t=top row, m=middle row, b=bottom row; 0=left, 1=center, 2=right)\n\n"
    f"Stage 2 - Squared/cross products (int16, scale={SOBEL_SCALE}):\n"
    f"    Ixx = (|gx| * |gx|) >> {SOBEL_SCALE}\n"
    f"    Iyy = (|gy| * |gy|) >> {SOBEL_SCALE}\n"
    f"    Ixy = (gx * gy) >> {SOBEL_SCALE}\n\n"
    "Stage 3 - Box filter 3x3 (normalized, int16):\n"
    f"    Sxx = (sum9(Ixx) * {XF_DIV_VAL_3x3}) >> 15  (approx 1/9)\n"
    f"    Syy = (sum9(Iyy) * {XF_DIV_VAL_3x3}) >> 15\n"
    f"    Sxy = (sum9(Ixy) * {XF_DIV_VAL_3x3}) >> 15\n\n"
    "Stage 4 - Harris score (int32):\n"
    "    a = Sxx >> 2,  b = Syy >> 2,  c = Sxy >> 2\n"
    "    det   = a * b - c * c\n"
    f"    score = det - ((a+b)^2 * {K_FIXED}) >> 16\n\n"
    "Stage 5 - Threshold:\n"
    f"    thresh_val = score if score > {THRESHOLD} else 0\n\n"
    "Stage 6 - Non-maximum suppression (4-neighbor, NMS_RADIUS=1):\n"
    "    Output 255 if thresh_val > top AND thresh_val > left AND\n"
    "                  thresh_val > right AND thresh_val > bottom\n"
    "    else 0\n"
)

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
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "out",
    },
]


def behavioral(img_in):
    img = img_in.reshape((128, 128))
    harris = cv2.cornerHarris(img.astype(np.float32), blockSize=3, ksize=3, k=0.04)
    out = np.zeros((128, 128), dtype=np.uint8)
    out[harris > 0.01 * harris.max()] = 255
    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
