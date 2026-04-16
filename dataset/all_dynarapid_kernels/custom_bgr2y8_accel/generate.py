import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 1080
WIDTH = 1920

function_signature = (
    "void custom_bgr2y8_accel("
    "int32_t* src, "
    "int32_t* dst, "
    "int32_t* array_params)"
)

description = f"""Custom BGR-to-Y8 kernel that converts a {WIDTH}x{HEIGHT} 3-channel 8-bit BGR image
to a single-channel 8-bit grayscale image using HSV-based color classification.

Parameters:
    src:          Pointer to flat array of {HEIGHT * WIDTH * 3} uint8 values representing
                  a {HEIGHT}x{WIDTH} BGR image in row-major, channel-interleaved order
                  (B0,G0,R0,B1,G1,R1,...).
    dst:          Pointer to flat array of {HEIGHT * WIDTH} uint8 output values.
    array_params: Pointer to array of 12 uint8 threshold parameters:
                  [0] black_Vmax=20, [1] black_Smax=70, [2] brown_Hmax=15,
                  [3] brown_Vmax=40, [4] Smin=60, [5] Smax=90,
                  [6] darkgreen_Vmax=35, [7] darkgreen_Hmin=30, [8] darkgreen_Hmax=45,
                  [9] green_Hmax=90, [10] green_Hmin=50, [11] green_Vmax=45.

Algorithm:
    1. Convert each BGR pixel to HSV using integer arithmetic:
       - V = max(B, G, R)
       - S = 255 * (V - min(B, G, R)) / V  (0 if V=0)
       - H = hue angle in [0, 180) following OpenCV convention
    2. Compute percentage values: V_per = (V * 100) / 255, sat_per = (S * 100) / 255
    3. Convert H from OpenCV's [0,180) range to degrees: H_degree = H * 2
    4. Classify each pixel:
       - If V_per < black_Vmax AND sat_per < black_Smax: output = 0 (black/background)
       - Else if H_degree < brown_Hmax AND V_per < brown_Vmax AND Smin < sat_per < Smax: output = 0 (brown defect)
       - Else if darkgreen_Hmin < H_degree < darkgreen_Hmax AND V_per < darkgreen_Vmax AND Smin < sat_per < Smax: output = 0 (dark green)
       - Else if green_Hmin < H_degree < green_Hmax AND V_per < green_Vmax: output = min(V + 30, 255) (green enhancement)
       - Else: output = V (pass-through)
"""

inputs_info = [
    {
        "name": "src",
        "array_dimension": 2,
        "shape": (HEIGHT * WIDTH, 3),
        "dtype": "int32",
        "total_elements": HEIGHT * WIDTH * 3,
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "dst",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "dtype": "int32",
        "total_elements": HEIGHT * WIDTH,
        "iotype": "out",
    },
    {
        "name": "array_params",
        "array_dimension": 1,
        "shape": (12,),
        "dtype": "int32",
        "total_elements": 12,
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
]

# Lookup tables from HLS bgr2hsv implementation
_HDIV = np.array([
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
], dtype=np.int64)

_SDIV = np.array([
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
], dtype=np.int64)


def _bgr2hsv(b, g, r):
    """Convert BGR to HSV using the same integer arithmetic as the HLS kernel.
    Inputs and outputs are numpy arrays of int64.
    Returns h, s, v arrays (each uint8 range)."""
    v = np.maximum(np.maximum(b, g), r)
    vmin = np.minimum(np.minimum(b, g), r)
    diff = v - vmin

    # Saturation
    s = np.where(v > 0,
                 (diff * _SDIV[v] + (1 << 11)) >> 12,
                 np.int64(0))

    # Hue
    vr = np.where(v == r, np.int64(-1), np.int64(0))
    vg = np.where(v == g, np.int64(-1), np.int64(0))

    h_raw = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))))
    # Need to handle diff=0 carefully for hdiv lookup
    safe_diff = np.clip(diff, 0, 255).astype(np.int64)
    h = (h_raw * _HDIV[safe_diff] + (1 << 11)) >> 12
    h = np.where(h < 0, h + 180, h)

    return h.astype(np.int64), s.astype(np.int64), v.astype(np.int64)


def behavioral(src, array_params):
    """Convert BGR image to Y8 using HSV-based classification."""
    # Parse params
    black_Vmax = int(array_params[0])
    black_Smax = int(array_params[1])
    brown_Hmax = int(array_params[2])
    brown_Vmax = int(array_params[3])
    Smin = int(array_params[4])
    Smax = int(array_params[5])
    darkgreen_Vmax = int(array_params[6])
    darkgreen_Hmin = int(array_params[7])
    darkgreen_Hmax = int(array_params[8])
    green_Hmax = int(array_params[9])
    green_Hmin = int(array_params[10])
    green_Vmax = int(array_params[11])

    # Parse BGR pixels
    pixels = src.reshape(-1, 3).astype(np.int64)
    b_ch = pixels[:, 0]
    g_ch = pixels[:, 1]
    r_ch = pixels[:, 2]

    # Convert BGR to HSV
    h, s, v = _bgr2hsv(b_ch, g_ch, r_ch)

    # Compute derived values as in HLS hsv2y8
    H_degree = h * 2
    V_per = (v * 100) // 255
    sat_per = (s * 100) // 255

    max_val = v.astype(np.int64)

    # Initialize output
    out = np.zeros(len(b_ch), dtype=np.uint8)

    # Classification (must follow exact HLS if-else order)
    # Condition 1: black/background
    is_black = (V_per < black_Vmax) & (sat_per < black_Smax)

    # Condition 2: brown defect
    is_brown = (~is_black) & (H_degree < brown_Hmax) & (V_per < brown_Vmax) & \
               (sat_per < Smax) & (sat_per > Smin)

    # Condition 3: dark green
    is_darkgreen = (~is_black) & (~is_brown) & \
                   (H_degree > darkgreen_Hmin) & (H_degree < darkgreen_Hmax) & \
                   (V_per < darkgreen_Vmax) & (sat_per < Smax) & (sat_per > Smin)

    # Condition 4: green enhancement
    is_green = (~is_black) & (~is_brown) & (~is_darkgreen) & \
               (H_degree > green_Hmin) & (H_degree < green_Hmax) & \
               (V_per < green_Vmax)

    # Other: everything else
    is_other = ~(is_black | is_brown | is_darkgreen | is_green)

    # Apply outputs
    out[is_black] = 0
    out[is_brown] = 0
    out[is_darkgreen] = 0
    out[is_green] = np.minimum(max_val[is_green] + 30, 255).astype(np.uint8)
    out[is_other] = max_val[is_other].astype(np.uint8)

    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
