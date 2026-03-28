# Example generate.py for: aec_accel
#
# Top function: aec_accel
# Source: /home/plwalker/Vitis_Libraries/vision/L1/examples/aec/
#
# Key decisions:
#   - Input: XF_8UC3 (3-channel BGR uint8 image) -> IN_TYPE = XF_8UC3
#   - Output: XF_8UC3 (3-channel BGR uint8 image) -> OUT_TYPE = XF_8UC3
#   - The algorithm performs automatic exposure correction on 3-channel images by:
#     1. Converting BGR to HSV
#     2. Extracting V (value/brightness) channel
#     3. Performing histogram equalization on V channel
#     4. Merging equalized V with original H and S
#     5. Converting back to BGR
#   - rows/cols are runtime parameters -> NOT in inputs_info
#   - Two histogram buffers are used internally (ping-pong) -> NOT exposed as parameters
#
# IMPORTANT: The Vitis Vision library uses hardware-optimized integer arithmetic and
# lookup tables for BGR<->HSV conversion, which differ slightly from standard formulas.
# This behavioral() uses a simplified numpy-based approximation that captures the
# algorithm's essence for AI training purposes.

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

The AEC algorithm enhances image brightness and contrast through the following five steps:

**Step 1: BGR to HSV Conversion**
Each BGR pixel (B,G,R) is converted to HSV color space using standard formulas:
- V (Value/Brightness) = max(R, G, B), scaled to [0, 255]
- S (Saturation) = (V - min(R,G,B)) / V if V > 0, else 0, scaled to [0, 255]  
- H (Hue) is computed based on which RGB channel achieves the maximum:
  * If R is max: H = 60 * ((G - B) / (V - min))
  * If G is max: H = 60 * ((B - R) / (V - min) + 2)
  * If B is max: H = 60 * ((R - G) / (V - min) + 4)
  The result is scaled to [0, 179] for uint8 representation (OpenCV convention).

**Step 2: Histogram Computation**
A 256-bin histogram is computed for the V channel, where hist[i] counts the number of
pixels with V value equal to i.

**Step 3: Histogram Equalization**
The V channel is equalized using cumulative distribution function (CDF) normalization:
1. Compute CDF[i] = sum(hist[0] + hist[1] + ... + hist[i]) for each i in [0, 255]
2. Find CDF_min = the smallest non-zero CDF value (i.e., CDF of the darkest pixel present)
3. For each pixel with V = v, compute the equalized value:
   new_v = floor((CDF[v] - CDF_min) * 255 / (total_pixels - CDF_min))
4. The result is truncated (floor division) to uint8 range [0, 255].

This redistribution spreads the pixel intensities across the full dynamic range [0, 255],
improving contrast in underexposed or overexposed images.

**Step 4: Channel Merge**
The equalized V channel (new_v) is merged with the original H and S channels to form
an equalized HSV image (H, S, new_v).

**Step 5: HSV to BGR Conversion**
The equalized HSV image is converted back to BGR using the inverse transformation:
1. Compute C = new_v * S (chroma)
2. Compute H' = H / 60 (H in [0, 6] range, dividing the hue circle into 6 sectors)
3. Compute X = C * (1 - |H' mod 2 - 1|) (intermediate value)
4. Compute m = new_v - C (match value to add to all channels)
5. Assign (R', G', B') based on which sector H' falls into:
   - Sector 0 (H' in [0,1)): (R',G',B') = (C, X, 0)
   - Sector 1 (H' in [1,2)): (R',G',B') = (X, C, 0)
   - Sector 2 (H' in [2,3)): (R',G',B') = (0, C, X)
   - Sector 3 (H' in [3,4)): (R',G',B') = (0, X, C)
   - Sector 4 (H' in [4,5)): (R',G',B') = (X, 0, C)
   - Sector 5 (H' in [5,6)): (R',G',B') = (C, 0, X)
6. Final RGB = (R' + m, G' + m, B' + m), scaled to [0, 255] and clamped.

**Division and Rounding**: All divisions in histogram equalization use integer division
(floor/truncation). The hardware implementation uses bit shifts (e.g., >>8 for divide by 256)
and lookup tables for efficiency; this produces results equivalent to floor division.

**Border Handling**: Not applicable — AEC is a per-pixel point operation; no spatial convolution.

**Output Format**: Each output pixel is stored as uint8 in BGR order. Intermediate
calculations use larger integer types (int16, int32) to prevent overflow, but final
results are truncated (not rounded) to uint8.

**Implementation Note**: The Vitis Vision hardware library uses optimized integer lookup
tables (hdiv, sdiv arrays) for color space conversions, which may produce slightly different
numerical results (typically ±1-2 LSB) compared to floating-point formulas. The behavioral
model in this file uses standard floating-point numpy operations for clarity and portability.
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

def behavioral(src):
    """
    Automatic Exposure Correction (AEC) behavioral model.
    
    Implements: BGR → HSV → Histogram Equalization on V → HSV → BGR
    
    Uses standard computer vision formulas with floating-point arithmetic for clarity.
    The hardware implementation uses integer arithmetic and lookup tables, which may
    produce slightly different results (typically within ±2 LSB).
    """
    # Reshape to (HEIGHT, WIDTH, 3) BGR image
    img_bgr = src.reshape((HEIGHT, WIDTH, 3))
    
    # Extract BGR channels and normalize to [0, 1]
    B = img_bgr[:, :, 0].astype(np.float32) / 255.0
    G = img_bgr[:, :, 1].astype(np.float32) / 255.0
    R = img_bgr[:, :, 2].astype(np.float32) / 255.0
    
    # --- Step 1: BGR to HSV Conversion ---
    V = np.maximum(np.maximum(R, G), B)
    min_val = np.minimum(np.minimum(R, G), B)
    C = V - min_val  # Chroma
    
    # Saturation: S = C / V (avoid division by zero)
    S = np.where(V > 1e-6, C / V, 0)
    
    # Hue: computed in degrees [0, 360), then scaled to [0, 179] for uint8
    H = np.zeros_like(V)
    mask = C > 1e-6  # Only compute H where chroma is non-zero
    
    mask_r = mask & (V == R)
    mask_g = mask & (V == G)
    mask_b = mask & (V == B)
    
    H[mask_r] = 60 * (((G[mask_r] - B[mask_r]) / C[mask_r]) % 6)
    H[mask_g] = 60 * (((B[mask_g] - R[mask_g]) / C[mask_g]) + 2)
    H[mask_b] = 60 * (((R[mask_b] - G[mask_b]) / C[mask_b]) + 4)
    
    # Scale to uint8 ranges: H → [0,179], S → [0,255], V → [0,255]
    H_uint8 = (H / 2.0).clip(0, 179).astype(np.uint8)
    S_uint8 = (S * 255).clip(0, 255).astype(np.uint8)
    V_uint8 = (V * 255).clip(0, 255).astype(np.uint8)
    
    # --- Steps 2 & 3: Histogram Equalization on V Channel ---
    # Compute 256-bin histogram
    hist, _ = np.histogram(V_uint8.flatten(), bins=256, range=(0, 256))
    
    # Compute cumulative distribution function (CDF)
    cdf = hist.cumsum()
    
    # Find minimum non-zero CDF value
    cdf_min = cdf[cdf > 0].min() if (cdf > 0).any() else 0
    
    # Normalize CDF: (cdf[v] - cdf_min) / (total_pixels - cdf_min) * 255
    total_pixels = HEIGHT * WIDTH
    denom = total_pixels - cdf_min
    if denom > 0:
        # Use floor division to match hardware truncation behavior
        cdf_norm = np.floor((cdf.astype(np.float64) - cdf_min) * 255.0 / denom).astype(np.uint8)
    else:
        # All pixels have the same value; no equalization possible
        cdf_norm = np.arange(256, dtype=np.uint8)
    
    # Apply equalization lookup
    V_eq = cdf_norm[V_uint8]
    
    # --- Steps 4 & 5: Merge and HSV to BGR Conversion ---
    # Scale back to [0, 1] for HSV to BGR math
    H_f = H_uint8.astype(np.float32) * 2.0  # [0,179] → [0,358]
    S_f = S_uint8.astype(np.float32) / 255.0
    V_f = V_eq.astype(np.float32) / 255.0
    
    # Compute intermediate values
    C_out = V_f * S_f
    H_60 = H_f / 60.0  # Divide hue into 6 sectors
    X = C_out * (1 - np.abs(H_60 % 2 - 1))
    m = V_f - C_out
    
    # Initialize output channels
    R_out = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    G_out = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    B_out = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    
    # Assign RGB' based on hue sector
    for sector in range(6):
        if sector == 0:  # H in [0, 60)
            mask = (H_60 >= 0) & (H_60 < 1)
            R_out[mask], G_out[mask], B_out[mask] = C_out[mask], X[mask], 0
        elif sector == 1:  # H in [60, 120)
            mask = (H_60 >= 1) & (H_60 < 2)
            R_out[mask], G_out[mask], B_out[mask] = X[mask], C_out[mask], 0
        elif sector == 2:  # H in [120, 180)
            mask = (H_60 >= 2) & (H_60 < 3)
            R_out[mask], G_out[mask], B_out[mask] = 0, C_out[mask], X[mask]
        elif sector == 3:  # H in [180, 240)
            mask = (H_60 >= 3) & (H_60 < 4)
            R_out[mask], G_out[mask], B_out[mask] = 0, X[mask], C_out[mask]
        elif sector == 4:  # H in [240, 300)
            mask = (H_60 >= 4) & (H_60 < 5)
            R_out[mask], G_out[mask], B_out[mask] = X[mask], 0, C_out[mask]
        else:  # sector == 5, H in [300, 360)
            mask = (H_60 >= 5) & (H_60 < 6)
            R_out[mask], G_out[mask], B_out[mask] = C_out[mask], 0, X[mask]
    
    # Add match value and scale to [0, 255]
    R_out = ((R_out + m) * 255).clip(0, 255).astype(np.uint8)
    G_out = ((G_out + m) * 255).clip(0, 255).astype(np.uint8)
    B_out = ((B_out + m) * 255).clip(0, 255).astype(np.uint8)
    
    # Reconstruct BGR image
    result = np.stack([B_out, G_out, R_out], axis=2)
    
    return result.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
