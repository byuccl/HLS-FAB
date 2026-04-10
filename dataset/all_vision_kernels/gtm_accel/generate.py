import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 168
WIDTH = 256
CHANNELS = 3

function_signature = (
    "void gtm_accel("
    "ap_uint<16>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""This HLS kernel applies global tone mapping (GTM) to a {HEIGHT}x{WIDTH} 3-channel 16-bit BGR image,
producing a {HEIGHT}x{WIDTH} 3-channel 8-bit BGR output.

Parameters:
- img_in: Flat array of {HEIGHT * WIDTH * CHANNELS} uint16 values representing a {HEIGHT}x{WIDTH} BGR image
  in row-major, channel-interleaved order (B0, G0, R0, B1, G1, R1, ...).
- img_out: Flat array of {HEIGHT * WIDTH * CHANNELS} uint8 values for the output BGR image.

Algorithm:
1. Convert BGR to XYZ color space (standard 3x3 matrix transform).
2. Compute log10 of each Y channel value. Track mean, max, and min of log10(Y).
3. For each pixel, compute tone mapping:
   a. K1 = 2.4 / (L_max - L_min), where L_max and L_min are the max and min of log10(Y).
   b. For each pixel: val = log10(Y) - mean; sigma_sq = (c1^2) / (2 * (L_max - L_min)^2);
      K2 = (1 - K1) * exp(-val^2 * sigma_sq) + K1;
      Y_out = exp(c2 * K2 * val + mean).
   c. X_out = (Y_out / Y) * X; Z_out = (Y_out / Y) * Z.
   d. Saturate X_out, Y_out, Z_out to uint8.
4. Merge mapped X, Y, Z channels and convert XYZ to BGR.

Default parameters: c1 = 3.0, c2 = 1.5 (encoded as unsigned int * 256 in the hardware).
Pixels with Y=0 use log10(Y)=epsilon (very small value).
"""

inputs_info = [
    {
        "name": "img_in",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH * CHANNELS,),
        "total_elements": HEIGHT * WIDTH * CHANNELS,
        "dtype": "uint16",
        "iotype": "in",
        "min": 0,
        "max": 65535,
    },
    {
        "name": "img_out",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH * CHANNELS,),
        "total_elements": HEIGHT * WIDTH * CHANNELS,
        "dtype": "uint8",
        "iotype": "out",
    },
]

# Fixed-point coefficients from xf_cvt_color_utils.hpp
# BGR to XYZ (16-bit): result = (R*coeff_R + G*coeff_G + B*coeff_B) >> 15, clamp [0,65535]
_CVT_X = np.array([13515, 11717, 5915], dtype=np.int64)   # X = R*13515 + G*11717 + B*5915
_CVT_Y = np.array([6969, 23434, 2364], dtype=np.int64)     # Y = R*6969  + G*23434 + B*2364
_CVT_Z = np.array([636, 3906, 31137], dtype=np.int64)      # Z = R*636   + G*3906  + B*31137

# XYZ to BGR (8-bit): result = (X*coeff_X + Y*coeff_Y + Z*coeff_Z) >> 13, clamp [0,255]
# Coefficients are stored as uint16 but cast to (short int), so some become negative
_CVT_R = np.array([26546, np.int16(np.uint16(52944)).item(), np.int16(np.uint16(61452)).item()], dtype=np.int64)
_CVT_G = np.array([np.int16(np.uint16(57596)).item(), 15368, 340], dtype=np.int64)
_CVT_B = np.array([456, np.int16(np.uint16(63864)).item(), 8662], dtype=np.int64)


def _bgr2xyz_16bit(b, g, r):
    """Convert BGR (uint16) to XYZ (uint16) using exact fixed-point coefficients."""
    R = r.astype(np.int64)
    G = g.astype(np.int64)
    B = b.astype(np.int64)

    X_raw = R * _CVT_X[0] + G * _CVT_X[1] + B * _CVT_X[2]
    Y_raw = R * _CVT_Y[0] + G * _CVT_Y[1] + B * _CVT_Y[2]
    Z_raw = R * _CVT_Z[0] + G * _CVT_Z[1] + B * _CVT_Z[2]

    X = np.clip(X_raw >> 15, 0, 65535).astype(np.uint16)
    Y = np.clip(Y_raw >> 15, 0, 65535).astype(np.uint16)
    Z = np.clip(Z_raw >> 15, 0, 65535).astype(np.uint16)
    return X, Y, Z


def _xyz2bgr_8bit(x, y, z):
    """Convert XYZ (uint8) to BGR (uint8) using exact fixed-point coefficients."""
    X = x.astype(np.int32)
    Y = y.astype(np.int32)
    Z = z.astype(np.int32)

    R_raw = X * _CVT_R[0] + Y * _CVT_R[1] + Z * _CVT_R[2]
    G_raw = X * _CVT_G[0] + Y * _CVT_G[1] + Z * _CVT_G[2]
    B_raw = X * _CVT_B[0] + Y * _CVT_B[1] + Z * _CVT_B[2]

    R = np.clip(R_raw >> 13, 0, 255).astype(np.uint8)
    G = np.clip(G_raw >> 13, 0, 255).astype(np.uint8)
    B = np.clip(B_raw >> 13, 0, 255).astype(np.uint8)
    return B, G, R


def behavioral(img_in):
    """Apply global tone mapping to a 16-bit 3-channel BGR image."""
    c1 = 3.0
    c2 = 1.5

    img = img_in.reshape((HEIGHT, WIDTH, CHANNELS))
    b = img[:, :, 0].astype(np.uint16)
    g = img[:, :, 1].astype(np.uint16)
    r = img[:, :, 2].astype(np.uint16)

    # BGR to XYZ using exact fixed-point math
    X, Y, Z = _bgr2xyz_16bit(b, g, r)

    # Compute log10(Y) for tone mapping (using float, as testbench does)
    Y_float = Y.astype(np.float64)
    Y_safe = np.where(Y_float > 0, Y_float, np.finfo(np.float64).tiny)
    logY = np.where(Y_float > 0, np.log10(Y_safe), 0.0)

    mean_logY = float(np.mean(logY))
    L_max = float(np.max(logY))
    L_min = float(np.min(logY))

    # Tone mapping
    d0 = L_max - L_min
    if d0 < 1e-10:
        d0 = 1e-10
    K1 = 2.4 / d0
    sigma_sq = (c1 * c1) / (2.0 * d0 * d0)

    val = logY - mean_logY
    exp_val = np.exp(-(val * val * sigma_sq))
    K2 = (1.0 - K1) * exp_val + K1
    out_val = np.exp(c2 * K2 * val + mean_logY)

    # Apply ratio to X and Z channels
    ratio = np.where(Y_float > 0, out_val / Y_safe, 0.0)
    X_mapped = np.clip(np.round(ratio * X.astype(np.float64)), 0, 255).astype(np.uint8)
    Y_mapped = np.clip(np.round(out_val), 0, 255).astype(np.uint8)
    Z_mapped = np.clip(np.round(ratio * Z.astype(np.float64)), 0, 255).astype(np.uint8)

    # XYZ to BGR using exact fixed-point math
    # merge(zmapped, ymapped, xmapped) actually puts xmapped in ch0, ymapped in ch1, zmapped in ch2
    # (merge reverses: third arg -> ch0, second -> ch1, first -> ch2)
    # xyz2bgr then treats ch0=X, ch1=Y, ch2=Z, which is the correct order
    B_out, G_out, R_out = _xyz2bgr_8bit(X_mapped, Y_mapped, Z_mapped)

    result = np.stack([B_out, G_out, R_out], axis=2)
    return result.flatten().astype(np.uint8)


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
