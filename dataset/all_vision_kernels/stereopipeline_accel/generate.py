import numpy as np
import cv2
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
    "ap_uint<8>* img_L, "
    "ap_uint<8>* img_R, "
    "ap_uint<16>* img_disp, "
    "float* cameraMA_l, "
    "float* cameraMA_r, "
    "float* distC_l, "
    "float* distC_r, "
    "float* irA_l, "
    "float* irA_r, "
    "int* bm_state_arr)"
)

description = f"""
This HLS kernel implements a complete stereo vision pipeline that computes a disparity map
from a pair of stereo images. The pipeline consists of three stages:

1. InitUndistortRectifyMapInverse: For each camera (left and right), compute undistortion
   and rectification maps (map_x, map_y) from camera matrix, distortion coefficients,
   and inverse rectification matrix. The maps encode source coordinates in Q24.8 fixed-point
   format (float_coord * 256 stored as int32).

2. Remap with bilinear interpolation: Apply the computed maps to warp each input image into
   rectified coordinates using bilinear interpolation with 5-bit fractional precision (1/32).
   Out-of-bounds pixels are set to 0.

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
        "dtype": "uint8",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_R",
        "iotype": "in",
        "array_dimension": 2,
        "dtype": "uint8",
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_disp",
        "iotype": "out",
        "array_dimension": 2,
        "dtype": "uint16",
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


def behavioral(img_L, img_R, cameraMA_l=None, cameraMA_r=None,
               distC_l=None, distC_r=None, irA_l=None, irA_r=None,
               bm_state_arr=None):
    H, W = 720, 1280
    left = img_L.reshape((H, W)).astype(np.uint8)
    right = img_R.reshape((H, W)).astype(np.uint8)

    # Use defaults if not provided
    if cameraMA_l is None:
        cameraMA_l = np.array(DEFAULT_CAMERA_MA_L, dtype=np.float32)
    if cameraMA_r is None:
        cameraMA_r = np.array(DEFAULT_CAMERA_MA_R, dtype=np.float32)
    if distC_l is None:
        distC_l = np.array(DEFAULT_DIST_C_L, dtype=np.float32)
    if distC_r is None:
        distC_r = np.array(DEFAULT_DIST_C_R, dtype=np.float32)
    if irA_l is None:
        irA_l = np.array(DEFAULT_IR_A_L, dtype=np.float32)
    if irA_r is None:
        irA_r = np.array(DEFAULT_IR_A_R, dtype=np.float32)

    K_l = cameraMA_l.astype(np.float64).reshape(3, 3)
    K_r = cameraMA_r.astype(np.float64).reshape(3, 3)
    D_l = distC_l.astype(np.float64).reshape(5)
    D_r = distC_r.astype(np.float64).reshape(5)
    R_l = irA_l.astype(np.float64).reshape(3, 3)
    R_r = irA_r.astype(np.float64).reshape(3, 3)

    # Compute undistort+rectify maps
    map1_l, map2_l = cv2.initUndistortRectifyMap(K_l, D_l, R_l, K_l, (W, H), cv2.CV_32FC1)
    map1_r, map2_r = cv2.initUndistortRectifyMap(K_r, D_r, R_r, K_r, (W, H), cv2.CV_32FC1)

    # Remap
    left_rect = cv2.remap(left, map1_l, map2_l, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    right_rect = cv2.remap(right, map1_r, map2_r, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # StereoBM
    sbm = cv2.StereoBM_create(numDisparities=48, blockSize=15)
    disp = sbm.compute(left_rect, right_rect)
    return disp.astype(np.uint16).flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
