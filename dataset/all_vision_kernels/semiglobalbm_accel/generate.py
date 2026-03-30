import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 720
WIDTH = 1280

WINDOW_SIZE = 5
TOTAL_DISPARITY = 64
PARALLEL_UNITS = 32
NUM_DIR = 4

SMALL_PENALTY = 20
LARGE_PENALTY = 40

function_signature = (
    "void semiglobalbm_accel("
    "ap_uint<8>* img_in_l, "
    "ap_uint<8>* img_in_r, "
    "ap_uint<8>* img_out)"
)

description = f"""
This HLS kernel computes a stereo disparity map using Semi-Global Block Matching (SGBM).

Parameters:
    img_in_l: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values representing the left
              grayscale stereo image in row-major order ({HEIGHT}x{WIDTH}).
    img_in_r: Pointer to a flat array of {HEIGHT * WIDTH} uint8 values representing the right
              grayscale stereo image in row-major order ({HEIGHT}x{WIDTH}).
    img_out:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 values for the output
              disparity map (same dimensions as inputs). Output values are raw integer
              disparities in [0, {TOTAL_DISPARITY - 1}] (no subpixel interpolation, no
              fixed-point scaling).

Algorithm:
1. Census Transform (5x5, XF_BORDER_CONSTANT):
   For each pixel (i, j), a 5x5 census transform is computed using a 24-bit descriptor.
   The 5x5 neighborhood is scanned in row-major order. For each of the 24 neighbors
   (all positions except the center [2][2]), bit (23 - idx) is set to 1 if neighbor < center.
   Out-of-bounds pixels are treated as 0. The result is a 24-bit descriptor (upper 24 bits
   of a 32-bit value; the hardware truncates to 24 bits before Hamming distance).

2. Initial Cost (Hamming distance):
   For each pixel (i, j) and disparity d in [0, {TOTAL_DISPARITY}):
     C(i, j, d) = popcount( census_left(i,j) XOR census_right(i, j-d) )
   If j-d < 0, census_right is treated as 0. Result is uint8 in [0, 24].

3. Semi-Global Optimization along {NUM_DIR} directions:
   Directions (r=0..3): (i,j-1), (i-1,j-1), (i-1,j), (i-1,j+1).
   Path costs Lr are maintained as uint8 (saturating at 255).
   For each direction r, pixel (i,j), disparity d:
     Lr(i,j,d) = C(i,j,d) + min(Lr(prev,d),
                                  Lr(prev,d-1) + P1,
                                  Lr(prev,d+1) + P1,
                                  min_k Lr(prev,k) + P2) - min_k Lr(prev,k)
   where P1={SMALL_PENALTY}, P2={LARGE_PENALTY}.
   Border cases (first pixel along each direction): Lr(i,j,d) = C(i,j,d).
   At disparity boundaries: Lr(prev,d=-1) = Lr(prev,d=NDISP) = 255 - P1 (so +P1 = 255).
   All intermediate values are clamped to uint8 [0, 255].

4. Cost Aggregation:
   S(i, j, d) = sum over all {NUM_DIR} directions of Lr(direction, i, j, d).
   Accumulated as uint16 (sum of {NUM_DIR} uint8 values, max = {NUM_DIR * 255}).

5. Disparity Selection:
   For each pixel, select disparity d* = argmin_d S(i,j,d).
   In case of ties, the smallest disparity index wins (first minimum wins in the scan).
   Output is uint8 raw disparity value (0 to {TOTAL_DISPARITY - 1}).
   No subpixel interpolation. No uniqueness ratio check.

Note: The HLS implementation uses 8-bit saturation for path costs and 16-bit accumulation
across directions. The first minimum in case of tie is used (lowest disparity index wins).
"""

inputs_info = [
    {
        "name": "img_in_l",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_in_r",
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


def _census_transform_5x5(img):
    """
    Compute 5x5 census transform matching xFComputeTransform5x5 exactly.
    Scans the 5x5 window row-major, skips center [2][2].
    Bit (23-idx) = 1 if neighbor < center. Returns uint32 (only lower 24 bits used).
    Border pixels treated as 0 (XF_BORDER_CONSTANT).
    """
    H, W = img.shape
    # Pad with zeros (constant border)
    padded = np.pad(img.astype(np.int32), 2, mode='constant', constant_values=0)
    result = np.zeros((H, W), dtype=np.uint32)
    idx = 0
    for di in range(5):
        for dj in range(5):
            if di == 2 and dj == 2:
                continue  # skip center
            # neighbor at offset (di-2, dj-2) from center
            neighbor = padded[di:di+H, dj:dj+W]
            center = img.astype(np.int32)
            bit = (neighbor < center).astype(np.uint32)
            result |= (bit << (23 - idx))
            idx += 1
    return result


def _hamming8(a, b):
    """Popcount of (a XOR b) for uint32 arrays, returns uint8."""
    xor = a ^ b
    # count bits
    xor = xor - ((xor >> 1) & np.uint32(0x55555555))
    xor = (xor & np.uint32(0x33333333)) + ((xor >> 2) & np.uint32(0x33333333))
    xor = (xor + (xor >> 4)) & np.uint32(0x0F0F0F0F)
    xor = (xor * np.uint32(0x01010101)) >> 24
    return xor.astype(np.uint8)


def _sgm_path_cost(C, prev_lr_slice, NDISP, P1_int, P2_int, MAXU8=255):
    """
    Vectorized SGM update for one step along a path direction.
    C: [N, NDISP] int32 - current pixel costs for N independent paths
    prev_lr_slice: [N, NDISP] int32 - previous path costs
    Returns new_lr: [N, NDISP] uint8
    """
    # lr_d: previous cost at same disparity [N, NDISP]
    lr_d = prev_lr_slice

    # lr_d-1: previous cost at d-1 (border: MAXU8 - P1 for d=0)
    lr_dp = np.empty_like(lr_d)
    lr_dp[:, 0] = MAXU8 - P1_int  # d=-1 border: 255-P1+P1=255
    lr_dp[:, 1:] = lr_d[:, :-1]

    # lr_d+1: previous cost at d+1 (border: MAXU8 - P1 for d=NDISP-1)
    lr_dn = np.empty_like(lr_d)
    lr_dn[:, -1] = MAXU8 - P1_int
    lr_dn[:, :-1] = lr_d[:, 1:]

    # min over all k: [N, 1]
    lr_min = lr_d.min(axis=1, keepdims=True)

    # 4-way min
    cand = np.minimum(
        np.minimum(lr_d, lr_dp + P1_int),
        np.minimum(lr_dn + P1_int, lr_min + P2_int)
    )

    val = C + cand - lr_min
    return np.clip(val, 0, MAXU8).astype(np.uint8)


def behavioral(img_in_l, img_in_r):
    H, W = HEIGHT, WIDTH
    NDISP = TOTAL_DISPARITY
    P1_int = int(SMALL_PENALTY)
    P2_int = int(LARGE_PENALTY)
    MAXU8 = 255

    left = img_in_l.reshape((H, W)).astype(np.uint8)
    right = img_in_r.reshape((H, W)).astype(np.uint8)

    # Step 1: Census transform (24-bit, stored in uint32)
    census_l = _census_transform_5x5(left)
    census_r = _census_transform_5x5(right)

    # Mask to 24 bits (matches hardware truncation ap_uint<24>)
    census_l = census_l & np.uint32(0xFFFFFF)
    census_r = census_r & np.uint32(0xFFFFFF)

    # Step 2: Initial cost C[H, W, NDISP] as int32 (hamming, max=24)
    C = np.zeros((H, W, NDISP), dtype=np.int32)
    for d in range(NDISP):
        if d == 0:
            right_shifted = census_r
        else:
            right_shifted = np.zeros((H, W), dtype=np.uint32)
            right_shifted[:, d:] = census_r[:, :W-d]
        C[:, :, d] = _hamming8(census_l, right_shifted).astype(np.int32)

    # Step 3: SGM optimization - 4 directions using vectorized inner loops
    # Lr shape: [4, H, W, NDISP] uint8
    Lr = np.zeros((4, H, W, NDISP), dtype=np.uint8)

    # Direction 0: left-to-right, prev = (i, j-1)
    # Rows are independent; for each row, iterate j=0..W-1
    Lr[0, :, 0, :] = np.clip(C[:, 0, :], 0, MAXU8).astype(np.uint8)  # border: all rows j=0
    for j in range(1, W):
        # N=H parallel paths, each at column j with prev at j-1
        prev = Lr[0, :, j-1, :].astype(np.int32)  # [H, NDISP]
        Lr[0, :, j, :] = _sgm_path_cost(C[:, j, :], prev, NDISP, P1_int, P2_int)

    # Direction 1: top-left to bottom-right, prev = (i-1, j-1)
    # Border: i==0 or j==0
    Lr[1, 0, :, :] = np.clip(C[0, :, :], 0, MAXU8).astype(np.uint8)  # i=0: all cols
    Lr[1, 1:, 0, :] = np.clip(C[1:, 0, :], 0, MAXU8).astype(np.uint8)  # j=0: all rows except 0
    # For interior pixels: iterate diagonals. We need to go row by row, but for each row i>0,
    # each pixel j>0 depends on (i-1, j-1). We can iterate i from 1..H-1 and vectorize over
    # the "shifted" column index: pixel (i,j) depends on (i-1, j-1).
    # We can iterate row by row and vectorize over interior columns:
    for i in range(1, H):
        # prev for (i, j) is at (i-1, j-1); valid for j=1..W-1
        prev = Lr[1, i-1, :W-1, :].astype(np.int32)  # [W-1, NDISP] from cols 0..W-2
        Lr[1, i, 1:, :] = _sgm_path_cost(C[i, 1:, :], prev, NDISP, P1_int, P2_int)

    # Direction 2: top-to-bottom, prev = (i-1, j)
    # Columns are independent; for each col, iterate i=0..H-1
    Lr[2, 0, :, :] = np.clip(C[0, :, :], 0, MAXU8).astype(np.uint8)  # border: i=0
    for i in range(1, H):
        prev = Lr[2, i-1, :, :].astype(np.int32)  # [W, NDISP]
        Lr[2, i, :, :] = _sgm_path_cost(C[i, :, :], prev, NDISP, P1_int, P2_int)

    # Direction 3: top-right to bottom-left, prev = (i-1, j+1)
    # Border: i==0 or j==W-1
    Lr[3, 0, :, :] = np.clip(C[0, :, :], 0, MAXU8).astype(np.uint8)  # i=0
    Lr[3, 1:, W-1, :] = np.clip(C[1:, W-1, :], 0, MAXU8).astype(np.uint8)  # j=W-1
    for i in range(1, H):
        # prev for (i, j) is at (i-1, j+1); valid for j=0..W-2
        prev = Lr[3, i-1, 1:, :].astype(np.int32)  # [W-1, NDISP] from cols 1..W-1
        Lr[3, i, :W-1, :] = _sgm_path_cost(C[i, :W-1, :], prev, NDISP, P1_int, P2_int)

    # Step 4: Cost aggregation (uint16)
    S = Lr.astype(np.uint16).sum(axis=0)  # [H, W, NDISP]

    # Step 5: Disparity selection (argmin, ties: lowest d wins)
    disp = np.argmin(S, axis=2).astype(np.uint8)

    return disp.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
