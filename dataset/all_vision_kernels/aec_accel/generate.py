# Example generate.py for: aec_accel
#
# Top function: aec_accel
# Source: /home/plwalker/Vitis_Libraries/vision/L1/examples/aec/
#
# Key decisions:
#   - Input: XF_8UC3 (3-channel BGR uint8 image) -> IN_TYPE = XF_8UC3
#   - Output: XF_8UC3 (3-channel BGR uint8 image) -> OUT_TYPE = XF_8UC3
#   - The algorithm performs automatic exposure correction on 3-channel images by:
#     1. Converting BGR to HSV using integer LUT-based arithmetic (bgr2hsv from xf_bgr2hsv.hpp)
#     2. Extracting V (value/brightness) channel
#     3. Performing histogram equalization on V channel (xFEqualize from xf_hist_equalize.hpp)
#     4. Merging equalized V with original H and S
#     5. Converting back to BGR (hsv2bgr from xf_cvt_color.hpp)
#   - rows/cols are runtime parameters -> NOT in inputs_info
#   - Two histogram buffers are used internally (ping-pong) -> NOT exposed as parameters
#
# IMPORTANT: The Vitis Vision library uses hardware-optimized integer arithmetic and
# lookup tables for BGR<->HSV conversion. This behavioral() exactly replicates those
# integer LUT-based computations.

import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128

function_signature = (
    "void aec_accel("
    "ap_uint<8>* src, "
    "ap_uint<8>* dst)"
)

description = f"""
This HLS kernel performs automatic exposure correction (AEC) on a 3-channel BGR color image.

Parameters:
    src: Pointer to a flat array of {HEIGHT * WIDTH * 3} uint8 pixel values representing
         a {HEIGHT}x{WIDTH} BGR color image in row-major order (B,G,R,B,G,R,...).
    dst: Pointer to a flat array of {HEIGHT * WIDTH * 3} uint8 pixel values for the
         exposure-corrected output image (same dimensions and format as input).

The AEC algorithm uses exact integer arithmetic matching the Vitis HLS implementation.

**Step 1: BGR to HSV Conversion (xf_bgr2hsv.hpp)**
Each BGR pixel uses integer lookup tables hdiv[256] and sdiv[256]:
  - B, G, R are uint8 values from input.
  - V = max(B, G, R)                    [uint8, range 0-255]
  - vmin = min(B, G, R)                 [uint8]
  - diff = V - vmin                     [uint8]
  - vr = (V == R) ? -1 : 0             [int, all-ones mask or 0]
  - vg = (V == G) ? -1 : 0             [int, all-ones mask or 0]
  - S = (diff * sdiv[V] + (1 << 11)) >> 12    [int, then cast to uint8]
    where sdiv[V] approximates (255 << 20) / V (saturation scale factor)
  - h_num = (vr & (G-B)) + (~vr & ((vg & (B-R + 2*diff)) + ((~vg) & (R-G + 4*diff))))
  - H = (h_num * hdiv[diff] + (1 << 11)) >> 12    [int, then adjusted]
    where hdiv[diff] approximates (30 << 12) / diff (hue scale factor)
  - H += (H < 0) ? 180 : 0             [wraps negative hue]
  - H, S, V are stored as uint8 in output order: ch0=H, ch1=S, ch2=V

**Step 2: Histogram Computation**
A 256-bin histogram is computed for the V channel (ch2 of the HSV image).

**Step 3: Histogram Equalization (xFEqualize from xf_hist_equalize.hpp)**
The V channel is equalized using the Vitis fixed-point CDF normalization:
  - total_pixels = HEIGHT * WIDTH
  - init_val = total_pixels - hist[0]   (pixels excluding those with V==0)
  - If init_val == 0: scale = 0
    Else: scale = (1 << 31) / init_val  (integer division)
  - scale1 = 255 * scale                (uint40)
  - cum_hist[0] = 0
  - For i = 1..255:
      temp_sum += hist[i]               (running sum of hist[1..i])
      sum = temp_sum * scale1 + (1 << 30)   (rounding: add 0x40000000)
      cum_hist[i] = sum >> 31           (final normalized value, uint8)
  - Apply: V_eq[pixel] = cum_hist[V[pixel]]

**Step 4: Channel Merge**
The equalized HSV image is formed as: ch0=V_eq, ch1=S, ch2=H
(xf::cv::merge(vimage_eq, simage, himage, imgHelper6) puts V_eq at bits[7:0],
S at bits[15:8], H at bits[23:16])

**Step 5: HSV to BGR Conversion (xfhsv2bgr from xf_cvt_color.hpp)**
Reads H=ch0 (bits[7:0]), S=ch1 (bits[15:8]), V=ch2 (bits[23:16]) from merged image.
Uses ap_fixed<28,9> (19 fractional bits) arithmetic:
  - hscale = 1/30.0  (ap_ufixed<20,1,AP_RND>: quantized to 19 frac bits with rounding)
  - s_scale = 1/255.0 (ap_ufixed<20,1,AP_RND>: quantized to 19 frac bits with rounding)
  - mul_scl = s_scale * S               (ap_fixed<28,9>)
  - H_scl = int(H * hscale)             (unsigned char: sector index 0-5)
  - h_fix = H * hscale - H_scl          (ap_fixed<28,9>: fractional part within sector)
  - tab[0] = V
  - tab[1] = V * (1 - mul_scl)
  - tab[2] = V * (1 - mul_scl * h_fix)
  - tab[3] = V * (1 - mul_scl + mul_scl * h_fix)
  - sector_data[6][3] = {{1,3,0}},{{1,0,2}},{{3,0,1}},{{0,2,1}},{{0,1,3}},{{2,1,0}}
  - b_out = (unsigned char)(tab[sector_data[H_scl][0]])
  - g_out = (unsigned char)(tab[sector_data[H_scl][1]])
  - r_out = (unsigned char)(tab[sector_data[H_scl][2]])

**Output Format**: Each output pixel is stored as uint8 in BGR order (B,G,R).

**Integer Arithmetic Note**: The hdiv and sdiv arrays use pre-computed fixed-point
reciprocals scaled by (1<<12) or (1<<20), enabling efficient integer multiplication
followed by right-shift instead of division.
"""

inputs_info = [
    {
        "name": "src",
        "array_dimension": 3,
        "shape": (HEIGHT, WIDTH, 3),
        "total_elements": HEIGHT * WIDTH * 3,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "dst",
        "array_dimension": 3,
        "shape": (HEIGHT, WIDTH, 3),
        "total_elements": HEIGHT * WIDTH * 3,
        "dtype": "uint8",
        "iotype": "out",
    },
]

# hdiv[256] from xf_bgr2hsv.hpp: approximates (30 << 12) / diff for diff in [1,255]
# hdiv[0] = 0 (unused, diff=0 means achromatic)
_hdiv = [
    0,    122880, 61440, 40960, 30720, 24576, 20480, 17554, 15360, 13653, 12288, 11171, 10240, 9452, 8777, 8192,
    7680, 7228,   6827,  6467,  6144,  5851,  5585,  5343,  5120,  4915,  4726,  4551,  4389,  4237, 4096, 3964,
    3840, 3724,   3614,  3511,  3413,  3321,  3234,  3151,  3072,  2997,  2926,  2858,  2793,  2731, 2671, 2614,
    2560, 2508,   2458,  2409,  2363,  2318,  2276,  2234,  2194,  2156,  2119,  2083,  2048,  2014, 1982, 1950,
    1920, 1890,   1862,  1834,  1807,  1781,  1755,  1731,  1707,  1683,  1661,  1638,  1617,  1596, 1575, 1555,
    1536, 1517,   1499,  1480,  1463,  1446,  1429,  1412,  1396,  1381,  1365,  1350,  1336,  1321, 1307, 1293,
    1280, 1267,   1254,  1241,  1229,  1217,  1205,  1193,  1182,  1170,  1159,  1148,  1138,  1127, 1117, 1107,
    1097, 1087,   1078,  1069,  1059,  1050,  1041,  1033,  1024,  1016,  1007,  999,   991,   983,  975,  968,
    960,  953,    945,   938,   931,   924,   917,   910,   904,   897,   890,   884,   878,   871,  865,  859,
    853,  847,    842,   836,   830,   825,   819,   814,   808,   803,   798,   793,   788,   783,  778,  773,
    768,  763,    759,   754,   749,   745,   740,   736,   731,   727,   723,   719,   714,   710,  706,  702,
    698,  694,    690,   686,   683,   679,   675,   671,   668,   664,   661,   657,   654,   650,  647,  643,
    640,  637,    633,   630,   627,   624,   621,   617,   614,   611,   608,   605,   602,   599,  597,  594,
    591,  588,    585,   582,   580,   577,   574,   572,   569,   566,   564,   561,   559,   556,  554,  551,
    549,  546,    544,   541,   539,   537,   534,   532,   530,   527,   525,   523,   521,   518,  516,  514,
    512,  510,    508,   506,   504,   502,   500,   497,   495,   493,   492,   490,   488,   486,  484,  482,
]

# sdiv[256] from xf_bgr2hsv.hpp: approximates (255 << 20) / V for V in [1,255]
# sdiv[0] = 0 (V=0 -> S=0)
_sdiv = [
    0,     1044480, 522240, 348160, 261120, 208896, 174080, 149211, 130560, 116053, 104448, 94953, 87040, 80345,
    74606, 69632,   65280,  61440,  58027,  54973,  52224,  49737,  47476,  45412,  43520,  41779, 40172, 38684,
    37303, 36017,   34816,  33693,  32640,  31651,  30720,  29842,  29013,  28229,  27486,  26782, 26112, 25475,
    24869, 24290,   23738,  23211,  22706,  22223,  21760,  21316,  20890,  20480,  20086,  19707, 19342, 18991,
    18651, 18324,   18008,  17703,  17408,  17123,  16846,  16579,  16320,  16069,  15825,  15589, 15360, 15137,
    14921, 14711,   14507,  14308,  14115,  13926,  13743,  13565,  13391,  13221,  13056,  12895, 12738, 12584,
    12434, 12288,   12145,  12006,  11869,  11736,  11605,  11478,  11353,  11231,  11111,  10995, 10880, 10768,
    10658, 10550,   10445,  10341,  10240,  10141,  10043,  9947,   9854,   9761,   9671,   9582,  9495,  9410,
    9326,  9243,    9162,   9082,   9004,   8927,   8852,   8777,   8704,   8632,   8561,   8492,  8423,  8356,
    8290,  8224,    8160,   8097,   8034,   7973,   7913,   7853,   7795,   7737,   7680,   7624,  7569,  7514,
    7461,  7408,    7355,   7304,   7253,   7203,   7154,   7105,   7057,   7010,   6963,   6917,  6872,  6827,
    6782,  6739,    6695,   6653,   6611,   6569,   6528,   6487,   6447,   6408,   6369,   6330,  6292,  6254,
    6217,  6180,    6144,   6108,   6073,   6037,   6003,   5968,   5935,   5901,   5868,   5835,  5803,  5771,
    5739,  5708,    5677,   5646,   5615,   5585,   5556,   5526,   5497,   5468,   5440,   5412,  5384,  5356,
    5329,  5302,    5275,   5249,   5222,   5196,   5171,   5145,   5120,   5095,   5070,   5046,  5022,  4998,
    4974,  4950,    4927,   4904,   4881,   4858,   4836,   4813,   4791,   4769,   4748,   4726,  4705,  4684,
    4663,  4642,    4622,   4601,   4581,   4561,   4541,   4522,   4502,   4483,   4464,   4445,  4426,  4407,
    4389,  4370,    4352,   4334,   4316,   4298,   4281,   4263,   4246,   4229,   4212,   4195,  4178,  4161,
    4145,  4128,    4112,   4096,
]

hdiv = np.array(_hdiv, dtype=np.int64)
sdiv = np.array(_sdiv, dtype=np.int64)

# sector_data from xfhsv2bgr: maps HSV sector to tab[] indices for (b, g, r)
_sector_data = [[1, 3, 0], [1, 0, 2], [3, 0, 1], [0, 2, 1], [0, 1, 3], [2, 1, 0]]

# ap_ufixed<20,1,AP_RND>: 19 fractional bits, rounded (not truncated)
# hscale = 1/30.0, quantized with rounding to 19 frac bits
_hscale_frac = int(round((1.0 / 30.0) * (1 << 19)))  # = round(17476.267) = 17476
_hscale = _hscale_frac / (1 << 19)

# s_scale = 1/255.0, quantized with rounding to 19 frac bits
_sscale_frac = int(round((1.0 / 255.0) * (1 << 19)))  # = round(2056.8) = 2057
_sscale = _sscale_frac / (1 << 19)

# ap_fixed<28,9>: 19 fractional bits
_FP28_FRAC = 19
_FP28_SCALE = 1 << _FP28_FRAC


def fp28_9(x):
    """Quantize to ap_fixed<28,9>: 19 fractional bits, truncate toward negative infinity (floor)."""
    # ap_fixed truncates (rounds toward negative infinity in standard HLS)
    return np.floor(x * _FP28_SCALE).astype(np.int64)


def behavioral(src):
    """
    Automatic Exposure Correction (AEC) behavioral model.

    Exactly replicates the Vitis HLS implementation:
      BGR -> HSV (integer LUT arithmetic from xf_bgr2hsv.hpp)
      -> histogram equalization on V (xFEqualize fixed-point from xf_hist_equalize.hpp)
      -> HSV -> BGR (ap_fixed<28,9> arithmetic from xfhsv2bgr in xf_cvt_color.hpp)
    """
    # Reshape to (HEIGHT*WIDTH, 3) BGR image
    img_bgr = src.reshape((-1, 3)).astype(np.int64)
    B = img_bgr[:, 0]
    G = img_bgr[:, 1]
    R = img_bgr[:, 2]
    N = len(B)

    # --- Step 1: BGR to HSV using integer LUT arithmetic (xf_bgr2hsv.hpp) ---
    # v = max(B, G, R), vmin = min(B, G, R)
    V = np.maximum(np.maximum(B, G), R)
    vmin = np.minimum(np.minimum(B, G), R)
    diff = V - vmin  # chroma, uint8

    # vr = (V == R) ? -1 : 0  (bitwise: all-ones int or 0)
    # vg = (V == G) ? -1 : 0
    vr = np.where(V == R, np.int64(-1), np.int64(0))
    vg = np.where(V == G, np.int64(-1), np.int64(0))

    # S = (diff * sdiv[V] + (1 << 11)) >> 12
    # For V==0, sdiv[0]=0, so S=0
    S_int = (diff * sdiv[V] + (1 << 11)) >> 12
    S_uint8 = np.clip(S_int, 0, 255).astype(np.uint8)

    # H numerator:
    # h_num = (vr & (G-B)) + (~vr & ((vg & (B-R + 2*diff)) + ((~vg) & (R-G + 4*diff))))
    # All operations on int32/int64
    term_r = vr & (G - B)
    term_g = vg & (B - R + 2 * diff)
    term_b = (~vg) & (R - G + 4 * diff)
    term_not_r = (~vr) & (term_g + term_b)
    h_num = term_r + term_not_r

    # H = (h_num * hdiv[diff] + (1 << 11)) >> 12
    # For diff==0, hdiv[0]=0, so H=0 (achromatic)
    H_int = (h_num * hdiv[diff] + (1 << 11)) >> 12
    # H += (H < 0) ? 180 : 0
    H_int = H_int + np.where(H_int < 0, np.int64(180), np.int64(0))
    H_uint8 = np.clip(H_int, 0, 255).astype(np.uint8)

    # V stays as uint8
    V_uint8 = V.astype(np.uint8)

    # --- Steps 2 & 3: Histogram Equalization on V Channel (xFEqualize) ---
    # Compute 256-bin histogram of V
    hist = np.bincount(V_uint8, minlength=256).astype(np.int64)

    # Vitis fixed-point normalization:
    # init_val = total_pixels - hist[0]  (excludes pixels with V==0)
    total_pixels = N
    init_val = int(total_pixels - hist[0])

    if init_val == 0:
        scale = 0
    else:
        scale = (1 << 31) // init_val  # integer division

    scale1 = np.int64(255) * np.int64(scale)  # uint40 in HLS

    cum_hist = np.zeros(256, dtype=np.uint8)
    cum_hist[0] = 0
    temp_sum = np.int64(0)
    for i in range(1, 256):
        temp_sum += hist[i]
        s = temp_sum * scale1 + np.int64(0x40000000)  # add rounding offset
        cum_hist[i] = int(s >> 31) & 0xFF

    # Apply equalization lookup
    V_eq = cum_hist[V_uint8]

    # --- Step 4: Merge: result has V_eq at ch0 (bits[7:0]), S at ch1, H at ch2 ---
    # (xf::cv::merge(vimage_eq, simage, himage, imgHelper6))
    # xfChannelCombineKernel with 3 inputs puts:
    #   result bits[7:0]   = channel3 (himage = H)
    #   result bits[15:8]  = channel2 (simage = S)
    #   result bits[23:16] = channel1 (vimage_eq = V_eq)
    # So HSV2BGR reads: H=ch0=bits[7:0], S=ch1=bits[15:8], V=ch2=bits[23:16]
    H_hsv = H_uint8.astype(np.int64)
    S_hsv = S_uint8.astype(np.int64)
    V_hsv = V_eq.astype(np.int64)

    # --- Step 5: HSV to BGR (xfhsv2bgr from xf_cvt_color.hpp) ---
    # Uses ap_fixed<28,9> (19 frac bits)
    # hscale = ap_ufixed<20,1,AP_RND>(1/30.0), s_scale = ap_ufixed<20,1,AP_RND>(1/255.0)

    # mul_scl = s_scale * S  (ap_fixed<28,9>)
    # In fixed-point: mul_scl_fp = floor((_sscale_frac * S_hsv) / 2^19 * 2^19) = _sscale_frac * S_hsv
    # but result is ap_fixed<28,9> value = (_sscale_frac * S_hsv) >> 19 * 2^19 / 2^19
    # Effectively: mul_scl = (_sscale_frac * S_hsv) / 2^19  [as real number]
    # We keep everything in integer fixed-point (units of 1/2^19)
    mul_scl_fp = _sscale_frac * S_hsv  # integer, in units of 1/2^19

    # H_scl = (unsigned char)(H * hscale) - this is floor since cast to unsigned char
    # H * hscale = H_hsv * _hscale_frac / 2^19 [real]
    # (unsigned char) truncates to integer: floor(H_hsv * _hscale_frac / 2^19)
    H_times_hscale_fp = H_hsv * np.int64(_hscale_frac)  # in units of 1/2^19
    H_scl = (H_times_hscale_fp >> _FP28_FRAC).astype(np.int64)  # unsigned char
    if np.any(H_scl >= 6):
        H_scl = np.where(H_scl >= 6, H_scl - 6, H_scl)

    # h_fix = H * hscale - H_scl  (ap_fixed<28,9>, in [0,1))
    # h_fix_fp = H_times_hscale_fp - H_scl * 2^19  (in units of 1/2^19)
    h_fix_fp = H_times_hscale_fp - H_scl * _FP28_SCALE  # in units of 1/2^19

    # tab[0] = V  (ap_fixed<28,9>: just V as integer * 2^19 / 2^19)
    # tab values are in ap_fixed<28,9> (units of 1/2^19)
    V_fp = V_hsv * _FP28_SCALE  # V in units of 1/2^19

    # tab[1] = V * (1 - mul_scl)
    # = V * (1 - mul_scl_fp/2^19)
    # = V_fp/2^19 * (2^19 - mul_scl_fp) / 2^19 * 2^19
    # tab1_fp = floor(V_fp * (2^19 - mul_scl_fp) / 2^19)
    tab1_fp = (V_fp * (_FP28_SCALE - mul_scl_fp)) >> _FP28_FRAC

    # tab[2] = V * (1 - mul_scl * h_fix)
    # mul_scl * h_fix = (mul_scl_fp/2^19) * (h_fix_fp/2^19) [real]
    # = mul_scl_fp * h_fix_fp / 2^38  [real]
    # In ap_fixed<28,9>: mul_scl * h_fix = floor(mul_scl_fp * h_fix_fp / 2^19) [units 1/2^19]
    mul_hfix_fp = (mul_scl_fp * h_fix_fp) >> _FP28_FRAC
    # tab[2] = V * (1 - mul_hfix)
    # = floor(V_fp * (2^19 - mul_hfix_fp) / 2^19)
    tab2_fp = (V_fp * (_FP28_SCALE - mul_hfix_fp)) >> _FP28_FRAC

    # tab[3] = V * (1 - mul_scl + mul_scl * h_fix)
    # = V * (1 - mul_scl_fp/2^19 + mul_hfix_fp/2^19)
    # = floor(V_fp * (2^19 - mul_scl_fp + mul_hfix_fp) / 2^19)
    tab3_fp = (V_fp * (_FP28_SCALE - mul_scl_fp + mul_hfix_fp)) >> _FP28_FRAC

    # tab array as (N, 4) matrix (in units of 1/2^19)
    # tab[0] = V_fp, tab[1] = tab1_fp, tab[2] = tab2_fp, tab[3] = tab3_fp
    tab = np.stack([V_fp, tab1_fp, tab2_fp, tab3_fp], axis=1)  # shape (N, 4)

    # (unsigned char)(tab[x]) = truncate to integer part (floor), then clip to uint8
    # ap_fixed<28,9> integer cast: take bits[27:19] = tab_fp >> 19
    sector_data = np.array(_sector_data, dtype=np.int64)  # shape (6, 3)

    B_out = np.zeros(N, dtype=np.uint8)
    G_out = np.zeros(N, dtype=np.uint8)
    R_out = np.zeros(N, dtype=np.uint8)

    for sec in range(6):
        mask = (H_scl == sec)
        if not np.any(mask):
            continue
        idx_b = sector_data[sec, 0]
        idx_g = sector_data[sec, 1]
        idx_r = sector_data[sec, 2]

        m = np.where(mask)[0]
        b_val = tab[m, idx_b] >> _FP28_FRAC
        g_val = tab[m, idx_g] >> _FP28_FRAC
        r_val = tab[m, idx_r] >> _FP28_FRAC

        B_out[m] = np.clip(b_val, 0, 255).astype(np.uint8)
        G_out[m] = np.clip(g_val, 0, 255).astype(np.uint8)
        R_out[m] = np.clip(r_val, 0, 255).astype(np.uint8)

    # Reconstruct BGR image
    result = np.stack([B_out, G_out, R_out], axis=1)  # (N, 3)

    return result.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
