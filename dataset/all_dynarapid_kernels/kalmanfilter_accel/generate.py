import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
KF_N = 5   # Number of state variables
KF_M = 3   # Number of measurements
KF_C = 1   # Number of control inputs
# IN_TYPE = XF_32FC1 -> float32, ap_uint<32>
# OUT_TYPE = XF_32FC1 -> float32, ap_uint<32>
# INPUT_PTR_WIDTH = 32, OUTPUT_PTR_WIDTH = 32

# Control flag bits:
# bit0 = INIT_EN, bit1 = TIMEUPDATE_EN, bit2 = MEASUPDATE_EN
# bit3 = XOUT_EN_TU, bit4 = UDOUT_EN_TU
# bit5 = XOUT_EN_MU, bit6 = UDOUT_EN_MU
# Hardcoded control_flag = 1+2+4+32+64 = 103 (init + predict + correct, output MU results)
CONTROL_FLAG = 103

function_signature = (
    "void kalmanfilter_accel("
    "int32_t* in_A, "
    "int32_t* in_B, "
    "int32_t* in_Uq, "
    "int32_t* in_Dq, "
    "int32_t* in_H, "
    "int32_t* in_X0, "
    "int32_t* in_U0, "
    "int32_t* in_D0, "
    "int32_t* in_R, "
    "int32_t* in_u, "
    "int32_t* in_y, "
    "int32_t* out_X, "
    "int32_t* out_U, "
    "int32_t* out_D)"
)

description = f"""\
This HLS kernel implements a Kalman Filter with UD (upper-triangular/diagonal)
factored covariance representation. It performs initialization, time update
(prediction), and measurement update (correction) in a single call.

The filter operates with {KF_N} state variables, {KF_M} measurements, and
{KF_C} control input.

Parameters (all float32, stored as ap_uint<32> bit patterns):
    in_A:  State transition matrix, {KF_N}x{KF_N} = {KF_N*KF_N} elements, row-major.
    in_B:  Control input matrix, {KF_N}x{KF_C} = {KF_N*KF_C} elements, row-major.
    in_Uq: Upper-triangular factor of process noise covariance Q, {KF_N}x{KF_N} = {KF_N*KF_N} elements.
           Q = Uq * diag(Dq) * Uq^T. Upper-triangular with 1s on diagonal.
    in_Dq: Diagonal of process noise covariance factor, {KF_N} elements.
    in_H:  Measurement matrix, {KF_M}x{KF_N} = {KF_M*KF_N} elements, row-major.
    in_X0: Initial state vector, {KF_N} elements.
    in_U0: Upper-triangular factor of initial error covariance P0, {KF_N}x{KF_N} = {KF_N*KF_N} elements.
           P0 = U0 * diag(D0) * U0^T. Upper-triangular with 1s on diagonal.
    in_D0: Diagonal of initial error covariance factor, {KF_N} elements.
    in_R:  Measurement noise variance (diagonal), {KF_M} elements.
    in_u:  Control input vector, {KF_C} elements.
    in_y:  Measurement vector, {KF_M} elements.
    out_X: Output state vector after measurement update, {KF_N} elements.
    out_U: Output upper-triangular factor of updated error covariance, {KF_N}x{KF_N} = {KF_N*KF_N} elements.
    out_D: Output diagonal of updated error covariance factor, {KF_N} elements.

Algorithm:
    1. Reconstruct covariance: P0 = U0 * diag(D0) * U0^T, Q = Uq * diag(Dq) * Uq^T
    2. Time Update (Predict): X_pred = A * X0 + B * u, P_pred = A * P0 * A^T + Q
    3. Measurement Update (Correct):
       S = H * P_pred * H^T + diag(R)
       K = P_pred * H^T * S^(-1)
       X_post = X_pred + K * (y - H * X_pred)
       P_post = (I - K * H) * P_pred
    4. Factor P_post = U_out * diag(D_out) * U_out^T via modified Cholesky (UDU^T) decomposition.

All arithmetic is float32.
"""

inputs_info = [
    {
        "name": "in_A",
        "array_dimension": 2,
        "shape": (KF_N, KF_N),
        "total_elements": KF_N * KF_N,
        "dtype": "float32",
        "iotype": "in",
        "min": -2.0,
        "max": 2.0,
    },
    {
        "name": "in_B",
        "array_dimension": 2,
        "shape": (KF_N, KF_C),
        "total_elements": KF_N * KF_C,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 1.0,
    },
    {
        "name": "in_Uq",
        "array_dimension": 2,
        "shape": (KF_N, KF_N),
        "total_elements": KF_N * KF_N,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 1.0,
    },
    {
        "name": "in_Dq",
        "array_dimension": 1,
        "shape": (KF_N,),
        "total_elements": KF_N,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 1.0,
    },
    {
        "name": "in_H",
        "array_dimension": 2,
        "shape": (KF_M, KF_N),
        "total_elements": KF_M * KF_N,
        "dtype": "float32",
        "iotype": "in",
        "min": -0.001,
        "max": 0.001,
    },
    {
        "name": "in_X0",
        "array_dimension": 1,
        "shape": (KF_N,),
        "total_elements": KF_N,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 5.0,
    },
    {
        "name": "in_U0",
        "array_dimension": 2,
        "shape": (KF_N, KF_N),
        "total_elements": KF_N * KF_N,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 1.0,
    },
    {
        "name": "in_D0",
        "array_dimension": 1,
        "shape": (KF_N,),
        "total_elements": KF_N,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 1.0,
    },
    {
        "name": "in_R",
        "array_dimension": 1,
        "shape": (KF_M,),
        "total_elements": KF_M,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 0.01,
    },
    {
        "name": "in_u",
        "array_dimension": 1,
        "shape": (KF_C,),
        "total_elements": KF_C,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 10.0,
    },
    {
        "name": "in_y",
        "array_dimension": 1,
        "shape": (KF_M,),
        "total_elements": KF_M,
        "dtype": "float32",
        "iotype": "in",
        "min": 0.0,
        "max": 5.0,
    },
    {
        "name": "out_X",
        "array_dimension": 1,
        "shape": (KF_N,),
        "total_elements": KF_N,
        "dtype": "float32",
        "iotype": "out",
    },
    {
        "name": "out_U",
        "array_dimension": 2,
        "shape": (KF_N, KF_N),
        "total_elements": KF_N * KF_N,
        "dtype": "float32",
        "iotype": "out",
    },
    {
        "name": "out_D",
        "array_dimension": 1,
        "shape": (KF_N,),
        "total_elements": KF_N,
        "dtype": "float32",
        "iotype": "out",
    },
]


def _udu_decompose(P):
    """UDU^T decomposition of symmetric positive semi-definite matrix P.
    Returns U (unit upper triangular) and d (diagonal vector) such that P = U*diag(d)*U^T.
    All operations in float32 to match HLS."""
    n = P.shape[0]
    U = np.eye(n, dtype=np.float32)
    d = np.zeros(n, dtype=np.float32)

    for j in range(n - 1, -1, -1):
        d[j] = np.float32(P[j, j])
        if d[j] > np.float32(0.0):
            for i in range(j):
                U[i, j] = np.float32(P[i, j] / d[j])
            for i in range(j):
                for k in range(i + 1):
                    P[k, i] = np.float32(P[k, i] - U[i, j] * d[j] * U[k, j])
        else:
            for i in range(j):
                U[i, j] = np.float32(0.0)

    return U, d


def behavioral(in_A, in_B, in_Uq, in_Dq, in_H, in_X0, in_U0, in_D0, in_R, in_u, in_y):
    """Kalman filter: init + time update (predict) + measurement update (correct).
    Uses standard Kalman equations in float32, then UDU decomposes the result."""
    # Reshape inputs
    A = in_A.reshape((KF_N, KF_N)).astype(np.float32)
    B = in_B.reshape((KF_N, KF_C)).astype(np.float32)
    Uq = in_Uq.reshape((KF_N, KF_N)).astype(np.float32)
    Dq = in_Dq.reshape((KF_N,)).astype(np.float32)
    H = in_H.reshape((KF_M, KF_N)).astype(np.float32)
    X0 = in_X0.reshape((KF_N,)).astype(np.float32)
    U0 = in_U0.reshape((KF_N, KF_N)).astype(np.float32)
    D0 = in_D0.reshape((KF_N,)).astype(np.float32)
    R = in_R.reshape((KF_M,)).astype(np.float32)
    u = in_u.reshape((KF_C,)).astype(np.float32)
    y = in_y.reshape((KF_M,)).astype(np.float32)

    # Reconstruct covariance matrices from UD factors
    P0 = U0 @ np.diag(D0) @ U0.T
    Q = Uq @ np.diag(Dq) @ Uq.T

    # Time Update (Predict)
    X_pred = (A @ X0.reshape(-1, 1) + B @ u.reshape(-1, 1)).flatten().astype(np.float32)
    P_pred = (A @ P0 @ A.T + Q).astype(np.float32)

    # Measurement Update (Correct)
    S = (H @ P_pred @ H.T + np.diag(R)).astype(np.float32)
    S_inv = np.linalg.inv(S).astype(np.float32)
    K = (P_pred @ H.T @ S_inv).astype(np.float32)
    innovation = (y - H @ X_pred).astype(np.float32)
    X_post = (X_pred + K @ innovation).astype(np.float32)
    P_post = ((np.eye(KF_N, dtype=np.float32) - K @ H) @ P_pred).astype(np.float32)

    # Make P_post symmetric
    P_post = np.float32(0.5) * (P_post + P_post.T)

    # UDU^T decomposition
    U_out, D_out = _udu_decompose(P_post.copy())

    out_X = X_post.flatten().astype(np.float32)
    out_U = U_out.flatten().astype(np.float32)
    out_D = D_out.flatten().astype(np.float32)

    return out_X, out_U, out_D


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
