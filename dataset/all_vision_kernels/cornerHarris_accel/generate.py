import numpy as np
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
    "detector algorithm implemented in fixed-point arithmetic.\n\n"
    f"Parameters:\n"
    f"    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing\n"
    f"             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.\n"
    f"    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for\n"
    f"             the corner response map. Corner pixels are marked as 255; all others are 0.\n\n"
    f"The kernel uses the Harris corner detector with these FIXED parameters:\n"
    f"    FILTER_WIDTH = {FILTER_WIDTH} (Sobel kernel size)\n"
    f"    BLOCK_WIDTH  = {BLOCK_WIDTH} (box filter window size)\n"
    f"    NMS_RADIUS   = {NMS_RADIUS} (non-maximum suppression radius)\n"
    f"    threshold    = {THRESHOLD}  (FIXED absolute threshold on Harris score; NOT data-dependent)\n"
    f"    K            = {K} (Harris k factor, fixed-point: k={K_FIXED} = K * 2^16)\n\n"
    "IMPORTANT: The threshold is a FIXED value of 442 compared against the computed Harris score\n"
    "directly. It is NOT relative to the maximum score in the image.\n\n"
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
    f"    trace = a + b\n"
    f"    score = det - (trace * trace * {K_FIXED}) >> 16\n\n"
    "Stage 5 - Fixed threshold (value 442):\n"
    f"    thresh_val = score if score > {THRESHOLD} else 0\n"
    f"    (This uses a FIXED absolute threshold of {THRESHOLD}, NOT a relative/data-dependent threshold)\n\n"
    "Stage 6 - Non-maximum suppression (4-neighbor, NMS_RADIUS=1):\n"
    "    Output 255 if thresh_val > top AND thresh_val > left AND\n"
    "                  thresh_val > right AND thresh_val > bottom\n"
    "    else 0\n"
    "    (border pixels treat out-of-bounds neighbors as 0)\n"
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
    """Fixed-point Harris corner detector matching Vitis xf_harris.hpp exactly.

    Pipeline (FILTER_WIDTH=3, BLOCK_WIDTH=3, NMS_RADIUS=1, threshold=442, k=2621):
      Stage 1: Sobel 3x3 (BORDER_CONSTANT zero-pad) -> int16 gx, gy
      Stage 2: Squared/cross products >> SOBEL_SCALE=6 -> int16 Ixx, Iyy, Ixy
      Stage 3: Box filter 3x3 (BORDER_CONSTANT) * 3641 >> 15 -> int16 Sxx, Syy, Sxy
      Stage 4: Harris score = (a*b - c*c) - ((a+b)^2 * 2621) >> 16  where a=Sxx>>2 etc
      Stage 5: Threshold: score if score > 442 else 0
      Stage 6: NMS radius 1: output 255 if thresh > top AND > left AND > right AND > bottom
    """
    img = img_in.reshape((HEIGHT, WIDTH)).astype(np.int32)

    # --- Stage 1: Sobel 3x3 with BORDER_CONSTANT (zero-pad) ---
    padded = np.pad(img, 1, mode='constant', constant_values=0)
    # Sobel kernel (Vitis convention):
    # gx: [-1 0 +1; -2 0 +2; -1 0 +1]  (right - left)
    # gy: [-1 -2 -1;  0  0  0; +1 +2 +1] (bottom - top)
    t0 = padded[0:HEIGHT,   0:WIDTH]
    t1 = padded[0:HEIGHT,   1:WIDTH+1]
    t2 = padded[0:HEIGHT,   2:WIDTH+2]
    m0 = padded[1:HEIGHT+1, 0:WIDTH]
    m2 = padded[1:HEIGHT+1, 2:WIDTH+2]
    b0 = padded[2:HEIGHT+2, 0:WIDTH]
    b1 = padded[2:HEIGHT+2, 1:WIDTH+1]
    b2 = padded[2:HEIGHT+2, 2:WIDTH+2]

    gx = (t2 + b2 - t0 - b0) + 2 * (m2 - m0)
    gy = (b0 + b2 - t0 - t2) + 2 * (b1 - t1)
    gx = gx.astype(np.int16)
    gy = gy.astype(np.int16)

    # --- Stage 2: Squared/cross products with scale=6 (int16 output) ---
    gx32 = gx.astype(np.int32)
    gy32 = gy.astype(np.int32)
    abs_gx = np.abs(gx32)
    abs_gy = np.abs(gy32)
    Ixx = (abs_gx * abs_gx) >> SOBEL_SCALE
    Iyy = (abs_gy * abs_gy) >> SOBEL_SCALE
    Ixy = (gx32 * gy32) >> SOBEL_SCALE
    # clip to int16 range as HLS does
    Ixx = np.clip(Ixx, -32768, 32767).astype(np.int16)
    Iyy = np.clip(Iyy, -32768, 32767).astype(np.int16)
    Ixy = np.clip(Ixy, -32768, 32767).astype(np.int16)

    # --- Stage 3: Box filter 3x3 (BORDER_CONSTANT) * 3641 >> 15 -> int16 ---
    def box_filter_3x3(src):
        s = src.astype(np.int32)
        padded2 = np.pad(s, 1, mode='constant', constant_values=0)
        # 3x3 sum
        result = np.zeros_like(s)
        for dr in range(3):
            for dc in range(3):
                result += padded2[dr:dr+HEIGHT, dc:dc+WIDTH]
        result = (result * XF_DIV_VAL_3x3) >> 15
        return np.clip(result, -32768, 32767).astype(np.int16)

    Sxx = box_filter_3x3(Ixx)
    Syy = box_filter_3x3(Iyy)
    Sxy = box_filter_3x3(Ixy)

    # --- Stage 4: Harris score (int32) ---
    a = (Sxx.astype(np.int32)) >> 2
    b = (Syy.astype(np.int32)) >> 2
    c = (Sxy.astype(np.int32)) >> 2

    det_res = a * b - c * c
    trace_res = (a + b).astype(np.int64)
    trace_res1 = trace_res * trace_res
    trace_res2 = (trace_res1 * K_FIXED) >> 16

    score = (det_res.astype(np.int64) - trace_res2).astype(np.int32)

    # --- Stage 5: Threshold ---
    thresh = np.where(score > THRESHOLD, score, np.int32(0))

    # --- Stage 6: NMS radius 1 (4-neighbor: top, left, right, bottom) ---
    # Condition: thresh[r,c] > thresh[r-1,c] AND > thresh[r,c-1]
    #            AND > thresh[r+1,c] AND > thresh[r,c+1]
    # Border pixels have neighbor=0 (BORDER_CONSTANT behavior)
    th = thresh  # shape (H, W)
    # top neighbor (row-1), zero for row==0
    top = np.zeros_like(th)
    top[1:, :] = th[:HEIGHT-1, :]
    # bottom neighbor (row+1), zero for row==HEIGHT-1
    bot = np.zeros_like(th)
    bot[:HEIGHT-1, :] = th[1:, :]
    # left neighbor (col-1), zero for col==0
    left = np.zeros_like(th)
    left[:, 1:] = th[:, :WIDTH-1]
    # right neighbor (col+1), zero for col==WIDTH-1
    right = np.zeros_like(th)
    right[:, :WIDTH-1] = th[:, 1:]

    is_max = (th > top) & (th > left) & (th > right) & (th > bot)
    out = np.where(is_max, np.uint8(255), np.uint8(0)).astype(np.uint8)
    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
