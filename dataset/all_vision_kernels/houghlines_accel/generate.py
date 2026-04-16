"""
Generate script for Vitis Vision HoughLines kernel.

Parameters (from xf_config_params.h):
  HEIGHT=128, WIDTH=128
  RHOSTEP=1, THETASTEP=2, LINESMAX=512, DIAGVAL=2203
  MINTHETA=0, MAXTHETA=180
  IN_TYPE=XF_8UC1, INPUT_PTR_WIDTH=8

Top function: houghlines_accel
C++ signature:
  void houghlines_accel(ap_uint<8>* img_in, short threshold, short maxlines,
                        float* arrayy, float* arrayx, int rows, int cols)

behavioral() returns (arrayy_flat, arrayx_flat) as float32 arrays of shape (LINESMAX,).
"""

import numpy as np
import cv2

# ---- Kernel parameters ----
HEIGHT       = 128
WIDTH        = 128
RHOSTEP      = 1
THETASTEP    = 2       # in "6.1 format" — see hpp
LINESMAX     = 512
DIAGVAL      = 2203    # rhoN
MINTHETA     = 0
MAXTHETA     = 180

# AngleN = (2*(MAXTHETA-MINTHETA)) / THETASTEP = (2*180)/2 = 180
ANGLEN = (2 * (MAXTHETA - MINTHETA)) // THETASTEP  # 180

# From xf_houghlines.hpp constants
PAI_BY_360 = 0.008726   # pi/360
PAI_BY_180 = 0.017444   # pi/180

# function_signature used by the csim testbench generator
function_signature = (
    "void houghlines_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<32>* arrayy, "
    "ap_uint<32>* arrayx)"
)

inputs_info = [
    {
        "name": "img_in",
        "shape": (HEIGHT, WIDTH),
        "dtype": "uint8",
        "iotype": "in",
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
    {
        "name": "arrayy",
        "shape": (LINESMAX,),
        "dtype": "float32",
        "iotype": "out",
        "total_elements": LINESMAX,
    },
    {
        "name": "arrayx",
        "shape": (LINESMAX,),
        "dtype": "float32",
        "iotype": "out",
        "total_elements": LINESMAX,
    },
]


def behavioral(img_in_flat, threshold=20, maxlines=512):
    img = img_in_flat.reshape((128, 128)).astype(np.uint8)
    lines = cv2.HoughLines(img, 1, np.pi / 180, threshold)
    rho_out = np.zeros(512, dtype=np.float32)
    theta_out = np.zeros(512, dtype=np.float32)
    if lines is not None:
        n = min(len(lines), 512)
        for i in range(n):
            rho_out[i] = lines[i][0][0]
            theta_out[i] = lines[i][0][1]
    return rho_out, theta_out
