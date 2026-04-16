import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 720
WIDTH = 1280

XF_CAMERA_MATRIX_SIZE = 9
XF_DIST_COEFF_SIZE = 5
NO_OF_DISPARITIES = 48
PARALLEL_UNITS = 16
SAD_WINDOW_SIZE = 15
XF_REMAP_BUFSIZE = 128

# Default camera parameters (DANSCAMERA_HACKF_720P from cameraParameters.h)
DEFAULT_CAMERA_MA_L = np.array([933.173, 0.0, 663.451, 0.0, 933.173, 377.015, 0.0, 0.0, 1.0], dtype=np.float32)
DEFAULT_CAMERA_MA_R = np.array([933.467, 0.0, 678.297, 0.0, 933.467, 359.623, 0.0, 0.0, 1.0], dtype=np.float32)
DEFAULT_DIST_C_L = np.array([-0.169398, 0.0227329, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_DIST_C_R = np.array([-0.170581, 0.0249444, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_IR_A_L = np.array([0.0011976323, -0.0000000019, -0.8153011732,
                           0.0000000007, 0.0011976994, -0.4422348617,
                           0.0000126839, 0.0000001064, 0.9913820905], dtype=np.float32)
DEFAULT_IR_A_R = np.array([0.0011976994, 0.0, -0.8047567905,
                           -0.0, 0.0011976994, -0.4420566166,
                           -0.0, -0.0000001064, 1.0000392898], dtype=np.float32)

# Default BM state (from xFSBMState constructor with SAD_WINDOW_SIZE=15, NDISP=48, NDISP_UNIT=16)
# preFilterType=0, preFilterSize=15, preFilterCap=31, SADWindowSize=15,
# minDisparity=0, numberOfDisparities=48, textureThreshold=20, uniquenessRatio=15,
# ndisp_unit=16, sweepFactor=3, remainder=0
# Note: testbench overrides textureThreshold=20 and uniquenessRatio=15
DEFAULT_BM_STATE = np.array([0, 15, 31, 15, 0, 48, 20, 15, 16, 3, 0], dtype=np.int32)

function_signature = (
    "void stereopipeline_accel("
    "int32_t* img_L, "
    "int32_t* img_R, "
    "int32_t* img_disp, "
    "float* cameraMA_l, "
    "float* cameraMA_r, "
    "float* distC_l, "
    "float* distC_r, "
    "float* irA_l, "
    "float* irA_r, "
    "int32_t* bm_state_arr)"
)

description = f"""
This HLS kernel implements a complete stereo vision pipeline that computes a disparity map
from a pair of stereo images. The pipeline consists of three stages:

1. InitUndistortRectifyMapInverse: For each camera (left and right), compute undistortion
   and rectification maps (map_x, map_y) from camera matrix, distortion coefficients,
   and inverse rectification matrix. The maps store source coordinates in Q24.8 fixed-point
   format as int32: map_val = int32((uint32_t)(float_coord * 256)).
   This is floor-truncation toward zero for positive coords (same as C cast via uint32_t).
   Camera parameters are first converted to ap_fixed<32,12> before use.

2. Remap with Q24.8 fixed-point bilinear interpolation (5-bit fractional precision, 1/32):
   For each destination pixel (i, j):
   a. Read map_x[i,j] and map_y[i,j] as int32 Q24.8 values.
   b. Compute x_int = map_x >> 8 (floor of source x coordinate, integer part).
      Compute y_int = map_y >> 8 (floor of source y coordinate, integer part).
   c. Compute 5-bit fractional parts:
      x_frac = (map_x & 0xFF) >> 3  (low 8 bits of Q24.8, scaled to 5-bit = divide by 8)
      y_frac = (map_y & 0xFF) >> 3
   d. Bilinear interpolation weights (ap_ufixed<5,0> precision = multiples of 1/32):
      iu = x_frac / 32,  iv = y_frac / 32
      k00 = (1 - iv) * (1 - iu),  k01 = (1 - iv) * iu
      k10 = iv * (1 - iu),         k11 = iv * iu
   e. Sample source pixels at (y_int, x_int), (y_int, x_int+1),
      (y_int+1, x_int), (y_int+1, x_int+1).
   f. Output = k00*d00 + k01*d01 + k10*d10 + k11*d11, truncated to uint8.
   g. If source coordinates are out of bounds (x_int < 0 or x_int >= W or
      y_int < 0 or y_int >= H based on x_fl <= W-1 and y_fl <= H-1 check), output 0.

3. StereoBM (Block Matching): Compute disparity using Sum of Absolute Differences (SAD) with
   Sobel prefiltering:
   a. Apply 3x3 Sobel-X filter to both rectified images.
   b. Clip Sobel output: values < -cap -> 0, values > cap -> 2*cap, else value + cap
      (where cap = preFilterCap = 31). Border pixels are set to 0 before clipping.
   c. For each pixel, compute SAD over a {SAD_WINDOW_SIZE}x{SAD_WINDOW_SIZE} window for
      disparities 0 to {NO_OF_DISPARITIES-1}.
   d. Select disparity with minimum SAD. Apply texture threshold and uniqueness ratio checks.
   e. Compute subpixel refinement: delta = ((p - n) << 8) / (p + n - 2*minsad + |p - n|)
      where p and n are SAD costs of neighboring disparities.
   f. Output = (best_disparity * 256 + delta + 15) >> 4 as uint16.
   g. Filtered pixels (border, low texture, non-unique) output 0.

Parameters:
    img_L:        Flat array of {HEIGHT * WIDTH} uint8 values, left grayscale image ({HEIGHT}x{WIDTH}).
    img_R:        Flat array of {HEIGHT * WIDTH} uint8 values, right grayscale image ({HEIGHT}x{WIDTH}).
    img_disp:     Flat array of {HEIGHT * WIDTH} uint16 values, output disparity map.
    cameraMA_l:   Array of {XF_CAMERA_MATRIX_SIZE} float32 values, left camera matrix (row-major 3x3).
    cameraMA_r:   Array of {XF_CAMERA_MATRIX_SIZE} float32 values, right camera matrix (row-major 3x3).
    distC_l:      Array of {XF_DIST_COEFF_SIZE} float32 values, left distortion coefficients (k1,k2,p1,p2,k3).
    distC_r:      Array of {XF_DIST_COEFF_SIZE} float32 values, right distortion coefficients.
    irA_l:        Array of {XF_CAMERA_MATRIX_SIZE} float32 values, left inverse rectification matrix (row-major 3x3).
    irA_r:        Array of {XF_CAMERA_MATRIX_SIZE} float32 values, right inverse rectification matrix.
    bm_state_arr: Array of 11 int32 values encoding StereoBM parameters:
                  [preFilterType, preFilterSize, preFilterCap, SADWindowSize, minDisparity,
                   numberOfDisparities, textureThreshold, uniquenessRatio, ndisp_unit, sweepFactor, remainder].
"""

inputs_info = [
    {
        "name": "img_L",
        "iotype": "in",
        "array_dimension": 2,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_R",
        "iotype": "in",
        "array_dimension": 2,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_disp",
        "iotype": "out",
        "array_dimension": 2,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
    },
    {
        "name": "cameraMA_l",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "float32",
        "shape": (XF_CAMERA_MATRIX_SIZE,),
        "total_elements": XF_CAMERA_MATRIX_SIZE,
    },
    {
        "name": "cameraMA_r",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "float32",
        "shape": (XF_CAMERA_MATRIX_SIZE,),
        "total_elements": XF_CAMERA_MATRIX_SIZE,
    },
    {
        "name": "distC_l",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "float32",
        "shape": (XF_DIST_COEFF_SIZE,),
        "total_elements": XF_DIST_COEFF_SIZE,
    },
    {
        "name": "distC_r",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "float32",
        "shape": (XF_DIST_COEFF_SIZE,),
        "total_elements": XF_DIST_COEFF_SIZE,
    },
    {
        "name": "irA_l",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "float32",
        "shape": (XF_CAMERA_MATRIX_SIZE,),
        "total_elements": XF_CAMERA_MATRIX_SIZE,
    },
    {
        "name": "irA_r",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "float32",
        "shape": (XF_CAMERA_MATRIX_SIZE,),
        "total_elements": XF_CAMERA_MATRIX_SIZE,
    },
    {
        "name": "bm_state_arr",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "int32",
        "shape": (11,),
        "total_elements": 11,
    },
]


def _compute_undistort_coords(cameraMatrix, distCoeffs, ir, rows, cols):
    """
    Compute undistortion+rectification map matching xFComputeUndistortCoordinates.
    All arithmetic in float32 (matching ap_fixed<32,12> precision).
    Returns (map_x_q24_8, map_y_q24_8) as int32 arrays of shape (rows, cols).
    map_val = int32((uint32_t)(float_coord * 256))  [Q24.8 truncation]
    """
    # Extract camera matrix elements
    fx = float(cameraMatrix[0])
    u0 = float(cameraMatrix[2])
    fy = float(cameraMatrix[4])
    v0 = float(cameraMatrix[5])

    k1 = float(distCoeffs[0])
    k2 = float(distCoeffs[1])
    p1 = float(distCoeffs[2])
    p2 = float(distCoeffs[3])
    k3 = float(distCoeffs[4])

    # ir is row-major 3x3 inverse rectification matrix
    # xFComputeUndistortCoordinates uses:
    #   _x = i*ir[1] + j*ir[0] + ir[2]
    #   _y = i*ir[4] + j*ir[3] + ir[5]
    #   w  = i*ir[7] + j*ir[6] + ir[8]
    # Note: ir layout is column-major-ish: ir[0]=R[0,1], ir[1]=R[0,0]?
    # Actually looking at the code: ir[0], ir[1], ir[2] are the first 3 elements
    # of the ir array passed in. The ir array is passed as a flat 9-element array.
    # _x = i*ir[1] + j*ir[0] + ir[2]
    # _y = i*ir[4] + j*ir[3] + ir[5]
    #  w = i*ir[7] + j*ir[6] + ir[8]

    i_idx = np.arange(rows, dtype=np.float64)
    j_idx = np.arange(cols, dtype=np.float64)
    jj, ii = np.meshgrid(j_idx, i_idx)

    ir = ir.astype(np.float64)

    _x = ii * ir[1] + jj * ir[0] + ir[2]
    _y = ii * ir[4] + jj * ir[3] + ir[5]
    w  = ii * ir[7] + jj * ir[6] + ir[8]

    # Perspective divide (xf_one_over_x_approx with ap_fixed uses 2-x approx,
    # but with float/float64 this is exact 1/w)
    winv = 1.0 / w
    x = _x * winv
    y = _y * winv

    x2 = x * x
    y2 = y * y
    xy2 = 2.0 * x * y
    r2 = x2 + y2

    kr = 1.0 + (k3 * r2 + k2) * r2 * r2 + k1 * r2
    # Simplified: kr = 1 + k1*r2 + k2*r2^2 + k3*r2^3
    kr = 1.0 + r2 * (k1 + r2 * (k2 + r2 * k3))

    u = fx * (x * kr + p1 * xy2 + p2 * (2.0 * x2 + r2)) + u0
    v = fy * (y * kr + p1 * (r2 + 2.0 * y2) + p2 * xy2) + v0

    # Q24.8 encoding: int32((uint32_t)(float * 256))
    # C cast (uint32_t)(float) truncates toward zero, then reinterpret as int32
    # For positive values: floor. For negative: truncate toward zero.
    # np.uint32(float) does the same truncation.
    with np.errstate(invalid='ignore', over='ignore'):
        map_x_q = np.int32(np.uint32(u * 256.0))
        map_y_q = np.int32(np.uint32(v * 256.0))

    return map_x_q, map_y_q


def _remap_q24_8(src, map_x_q, map_y_q):
    """
    Remap src using Q24.8 fixed-point maps with 5-bit fractional bilinear interpolation.
    map_x_q, map_y_q: int32 arrays of shape (H, W), Q24.8 encoded.
    Returns remapped image as uint8.
    """
    H, W = src.shape
    XF_RESIZE_INTER_BITS = 5
    XF_RESIZE_INTER_TAB_SIZE = 32  # 2^5

    # Integer parts (floor via arithmetic right shift of Q24.8)
    # x_int = map_x_q >> 8 (arithmetic right shift for int32)
    x_int = map_x_q >> 8  # numpy uses arithmetic shift for int32
    y_int = map_y_q >> 8

    # 5-bit fractional parts: low 8 bits of Q24.8, then scale to 5-bit
    # x_fl = (map_x_q & 0xFF) / 256  (sub-pixel fraction)
    # x_frac_5 = floor(x_fl * 32) = floor((map_x_q & 0xFF) / 8)
    x_low8 = map_x_q & np.int32(0xFF)
    y_low8 = map_y_q & np.int32(0xFF)
    # Handle negative low8 (when map_x_q is negative, & 0xFF still gives correct low byte)
    # But int32 & 0xFF in numpy: need to ensure we get unsigned low byte
    x_low8 = (map_x_q.astype(np.int32) & 0xFF).astype(np.int32)
    y_low8 = (map_y_q.astype(np.int32) & 0xFF).astype(np.int32)

    x_frac = x_low8 >> 3   # divide by 8 to get 5-bit frac
    y_frac = y_low8 >> 3

    # Bilinear weights (5-bit fixed-point, 1/32 precision)
    # iu = x_frac / 32, iv = y_frac / 32
    # Using integer arithmetic matching ap_ufixed<5,0>:
    # k00 = (32 - y_frac) * (32 - x_frac)
    # k01 = (32 - y_frac) * x_frac
    # k10 = y_frac * (32 - x_frac)
    # k11 = y_frac * x_frac
    # result = (k00*d00 + k01*d01 + k10*d10 + k11*d11) / 1024
    # The hardware uses ap_ufixed arithmetic so this is exact.

    iu = x_frac.astype(np.int32)  # 0..31
    iv = y_frac.astype(np.int32)  # 0..31

    # In-bounds check matching hardware:
    # in_range = (y[n] >= 0) and (x_fl[n] <= cols-1) and (x[n] >= 0) and (y_fl[n] <= rows-1)
    # x_fl = (map_x_q as ufixed >> 8) which for non-negative is x_int + x_frac/32
    # For the check, hardware uses x_fl <= W-1 and y_fl <= H-1.
    # x_fl <= W-1 means (map_x_q >> 8) <= W-1, and fractional part doesn't matter for <=
    # because x_fl = x_int + frac, if x_int <= W-1 then x_fl could be up to W-1+31/32 < W
    # So effective check: x_int >= 0 && x_int <= W-1 && y_int >= 0 && y_int <= H-1
    # (x_int+1 being out of bounds is handled by clamped reads or zero output)
    in_range = ((x_int >= 0) & (x_int <= W - 1) &
                (y_int >= 0) & (y_int <= H - 1))

    # Clamp coordinates for safe array indexing
    x0 = np.clip(x_int, 0, W - 1)
    y0 = np.clip(y_int, 0, H - 1)
    x1 = np.clip(x_int + 1, 0, W - 1)
    y1 = np.clip(y_int + 1, 0, H - 1)

    src_f = src.astype(np.int32)

    d00 = src_f[y0, x0]
    d01 = src_f[y0, x1]
    d10 = src_f[y1, x0]
    d11 = src_f[y1, x1]

    # Compute bilinear with integer arithmetic (1024 = 32*32)
    # Matching ap_ufixed multiplication
    k00 = (32 - iv) * (32 - iu)
    k01 = (32 - iv) * iu
    k10 = iv * (32 - iu)
    k11 = iv * iu

    result = (d00 * k00 + d01 * k01 + d10 * k10 + d11 * k11) >> 10  # divide by 1024

    out = np.where(in_range, result.astype(np.uint8), np.uint8(0))
    return out.astype(np.uint8)


def _sobel_x_3x3(img):
    """
    Apply 3x3 Sobel-X filter. Result is int16. Border pixels = 0.
    Sobel-X kernel: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
    padded has shape (H+2, W+2); interior result covers rows 1..H-2, cols 1..W-2.
    """
    H, W = img.shape
    f = img.astype(np.int32)
    result = np.zeros((H, W), dtype=np.int16)
    # Pad with zeros (border constant)
    padded = np.pad(f, 1, mode='constant', constant_values=0)
    # padded shape is (H+2, W+2)
    # For each interior pixel (i,j) with i in [1,H-1), j in [1,W-1):
    # Sobel_X = -p[i-1,j-1] + p[i-1,j+1] - 2*p[i,j-1] + 2*p[i,j+1] - p[i+1,j-1] + p[i+1,j+1]
    # In padded coords: pixel (i,j) in original -> padded[i+1, j+1]
    # Interior: i in [1, H-1], j in [1, W-1] in original -> padded[2:H, 2:W] to padded[1:H-1, 1:W-1]
    # Use vectorized slice on the padded array covering interior of original [1..H-2, 1..W-2]
    p = padded  # (H+2, W+2)
    # interior of result: result[1:H-1, 1:W-1] (rows/cols 1 to H-2 inclusive)
    result[1:H-1, 1:W-1] = (
        -p[0:H-2, 0:W-2].astype(np.int32) + p[0:H-2, 2:W].astype(np.int32)
        - 2*p[1:H-1, 0:W-2].astype(np.int32) + 2*p[1:H-1, 2:W].astype(np.int32)
        - p[2:H, 0:W-2].astype(np.int32) + p[2:H, 2:W].astype(np.int32)
    ).astype(np.int16)
    # Borders (row 0, row H-1, col 0, col W-1) remain 0
    return result


def _prefilter_sobel(img, cap):
    """
    Apply Sobel-X then clip: < -cap -> 0, > cap -> 2*cap, else val + cap.
    """
    sobel = _sobel_x_3x3(img)
    clipped = np.where(sobel < -cap, 0,
               np.where(sobel > cap, 2 * cap, sobel + cap)).astype(np.uint8)
    return clipped


def _stereo_bm(left_rect, right_rect, bm_state):
    """
    StereoBM matching xf::cv::StereoBM implementation.
    Returns uint16 disparity map.
    """
    H, W = left_rect.shape
    preFilterCap = int(bm_state[2])
    sad_win = int(bm_state[3])
    min_disp = int(bm_state[4])
    n_disp = int(bm_state[5])
    texture_thresh = int(bm_state[6])
    uniqueness_ratio = int(bm_state[7])

    half_win = sad_win // 2

    # Prefilter
    left_f = _prefilter_sobel(left_rect, preFilterCap)
    right_f = _prefilter_sobel(right_rect, preFilterCap)

    disp_out = np.zeros((H, W), dtype=np.uint16)

    # Compute SAD for all disparities using integral images for efficiency
    # For each disparity d, compute SAD over sad_win x sad_win window
    sad_maps = np.zeros((n_disp, H, W), dtype=np.int32)
    for d in range(n_disp):
        d_actual = d + min_disp
        # Shift right image left by d_actual
        right_shifted = np.zeros_like(right_f, dtype=np.int32)
        if d_actual >= 0:
            if d_actual < W:
                right_shifted[:, :W-d_actual] = right_f[:, d_actual:]
        else:
            d_neg = -d_actual
            if d_neg < W:
                right_shifted[:, d_neg:] = right_f[:, :W-d_neg]

        diff = np.abs(left_f.astype(np.int32) - right_shifted)

        # Box filter (SAD window) using cumsum
        integral = diff.cumsum(axis=0).cumsum(axis=1)
        # Pad integral image for easier computation
        pad_int = np.pad(integral, ((1, 0), (1, 0)), mode='constant')
        # SAD[i,j] = sum over [i-half:i+half+1, j-half:j+half+1]
        r1 = np.clip(np.arange(H) - half_win, 0, H)
        r2 = np.clip(np.arange(H) + half_win + 1, 0, H)
        c1 = np.clip(np.arange(W) - half_win, 0, W)
        c2 = np.clip(np.arange(W) + half_win + 1, 0, W)

        # Use the padded integral image
        pad_int2 = np.zeros((H+1, W+1), dtype=np.int64)
        pad_int2[1:, 1:] = diff.cumsum(axis=0).cumsum(axis=1)

        for i in range(H):
            row_top = max(0, i - half_win)
            row_bot = min(H, i + half_win + 1)
            for j in range(W):
                col_left = max(0, j - half_win)
                col_right = min(W, j + half_win + 1)
                sad_maps[d, i, j] = (
                    pad_int2[row_bot, col_right]
                    - pad_int2[row_top, col_right]
                    - pad_int2[row_bot, col_left]
                    + pad_int2[row_top, col_left]
                )

    # Vectorized approach instead:
    sad_maps2 = np.zeros((n_disp, H, W), dtype=np.int32)
    for d in range(n_disp):
        d_actual = d + min_disp
        right_shifted = np.zeros_like(right_f, dtype=np.int32)
        if d_actual >= 0:
            if d_actual < W:
                right_shifted[:, :W-d_actual] = right_f[:, d_actual:]
        else:
            d_neg = -d_actual
            if d_neg < W:
                right_shifted[:, d_neg:] = right_f[:, :W-d_neg]

        diff = np.abs(left_f.astype(np.int32) - right_shifted)

        pad_int = np.zeros((H+1, W+1), dtype=np.int64)
        pad_int[1:, 1:] = diff.cumsum(axis=0).cumsum(axis=1)

        row_top = np.maximum(0, np.arange(H) - half_win)
        row_bot = np.minimum(H, np.arange(H) + half_win + 1)
        col_left = np.maximum(0, np.arange(W) - half_win)
        col_right = np.minimum(W, np.arange(W) + half_win + 1)

        # Vectorized box filter
        A = pad_int[row_bot[:, None], col_right[None, :]]
        B = pad_int[row_top[:, None], col_right[None, :]]
        C = pad_int[row_bot[:, None], col_left[None, :]]
        D = pad_int[row_top[:, None], col_left[None, :]]
        sad_maps2[d] = (A - B - C + D).astype(np.int32)

    sad_maps = sad_maps2

    # Find minimum SAD disparity
    min_sad_idx = np.argmin(sad_maps, axis=0)  # [H, W]
    min_sad = np.min(sad_maps, axis=0)          # [H, W]

    # Border mask: pixels where SAD window extends to valid region
    border_mask = np.zeros((H, W), dtype=bool)
    border_mask[:half_win, :] = True
    border_mask[H-half_win:, :] = True
    border_mask[:, :half_win + min_disp + n_disp - 1] = True
    border_mask[:, W-half_win:] = True

    # Texture threshold check
    # Sum of absolute diff in the SAD window of the left filtered image
    left_sad_integral = np.zeros((H+1, W+1), dtype=np.int64)
    left_sad_integral[1:, 1:] = left_f.astype(np.int64).cumsum(axis=0).cumsum(axis=1)
    row_top_t = np.maximum(0, np.arange(H) - half_win)
    row_bot_t = np.minimum(H, np.arange(H) + half_win + 1)
    col_left_t = np.maximum(0, np.arange(W) - half_win)
    col_right_t = np.minimum(W, np.arange(W) + half_win + 1)
    texture = (left_sad_integral[row_bot_t[:, None], col_right_t[None, :]]
               - left_sad_integral[row_top_t[:, None], col_right_t[None, :]]
               - left_sad_integral[row_bot_t[:, None], col_left_t[None, :]]
               + left_sad_integral[row_top_t[:, None], col_left_t[None, :]]).astype(np.int32)
    texture_mask = texture < texture_thresh

    # Uniqueness ratio check
    unique_mask = np.zeros((H, W), dtype=bool)
    if uniqueness_ratio > 0:
        for d in range(n_disp):
            # Mask out the best and neighboring disparities
            is_best = (d == min_sad_idx)
            is_neighbor = ((d == min_sad_idx - 1) | (d == min_sad_idx + 1))
            not_best_or_neighbor = ~(is_best | is_neighbor)
            # Check if any non-neighboring disparity has cost <= min_sad * (1 + ratio/100)
            threshold = min_sad * (100 + uniqueness_ratio) // 100
            violates = not_best_or_neighbor & (sad_maps[d] <= threshold)
            unique_mask |= violates

    # Subpixel refinement
    d_best = min_sad_idx.astype(np.int32)
    minsad = min_sad.astype(np.int32)

    # Previous and next disparity costs
    d_prev = np.clip(d_best - 1, 0, n_disp - 1)
    d_next = np.clip(d_best + 1, 0, n_disp - 1)

    p_cost = sad_maps[d_prev, np.arange(H)[:, None], np.arange(W)[None, :]]
    n_cost = sad_maps[d_next, np.arange(H)[:, None], np.arange(W)[None, :]]

    # At boundaries use mirror
    at_d0 = (d_best == 0)
    at_dmax = (d_best == n_disp - 1)
    p_cost = np.where(at_d0, n_cost, p_cost)
    n_cost = np.where(at_dmax, p_cost, n_cost)

    # delta = ((p - n) << 8) / (p + n - 2*minsad + |p - n|)
    num = (p_cost - n_cost) << 8
    denom = p_cost + n_cost - 2 * minsad + np.abs(p_cost - n_cost)
    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        delta = np.where(denom != 0, num // denom, 0).astype(np.int32)

    # Output = (best_disparity * 256 + delta + 15) >> 4
    out_val = ((d_best * 256 + delta + 15) >> 4).astype(np.int32)

    # Apply masks: filtered pixels output 0
    filtered = border_mask | texture_mask | unique_mask
    disp_out = np.where(filtered, 0, np.clip(out_val, 0, 65535)).astype(np.uint16)

    return disp_out


def behavioral(img_L, img_R, cameraMA_l=None, cameraMA_r=None,
               distC_l=None, distC_r=None, irA_l=None, irA_r=None,
               bm_state_arr=None):
    H, W = HEIGHT, WIDTH
    left = img_L.reshape((H, W)).astype(np.uint8)
    right = img_R.reshape((H, W)).astype(np.uint8)

    # Use defaults if not provided
    if cameraMA_l is None:
        cameraMA_l = DEFAULT_CAMERA_MA_L.copy()
    if cameraMA_r is None:
        cameraMA_r = DEFAULT_CAMERA_MA_R.copy()
    if distC_l is None:
        distC_l = DEFAULT_DIST_C_L.copy()
    if distC_r is None:
        distC_r = DEFAULT_DIST_C_R.copy()
    if irA_l is None:
        irA_l = DEFAULT_IR_A_L.copy()
    if irA_r is None:
        irA_r = DEFAULT_IR_A_R.copy()
    if bm_state_arr is None:
        bm_state_arr = DEFAULT_BM_STATE.copy()

    # Step 1: Compute Q24.8 maps
    map_xl_q, map_yl_q = _compute_undistort_coords(cameraMA_l, distC_l, irA_l, H, W)
    map_xr_q, map_yr_q = _compute_undistort_coords(cameraMA_r, distC_r, irA_r, H, W)

    # Step 2: Remap with Q24.8 bilinear interpolation
    left_rect = _remap_q24_8(left, map_xl_q, map_yl_q)
    right_rect = _remap_q24_8(right, map_xr_q, map_yr_q)

    # Step 3: StereoBM
    disp = _stereo_bm(left_rect, right_rect, bm_state_arr)

    return disp.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
