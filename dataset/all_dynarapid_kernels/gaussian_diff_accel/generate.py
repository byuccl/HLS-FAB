import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH_1 = 3   # first Gaussian filter size (FILTER_SIZE_1_3 = 1)
FILTER_WIDTH_2 = 3   # second Gaussian filter size (FILTER_SIZE_2_3 = 1)
# IN_TYPE = XF_8UC1, OUT_TYPE = XF_8UC1

# sigma1 and sigma2 are runtime float parameters; hardcoded here.
SIGMA1 = 1.0
SIGMA2 = 2.0

function_signature = (
    "void gaussian_diff_accel("
    "int32_t* img_in, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel computes a Difference of Gaussians (DoG) on a grayscale image using
fixed-point arithmetic that exactly matches the Vitis xf::cv::GaussianBlur and
xf::cv::subtract implementations.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the DoG output image (same dimensions as input).

The kernel applies the following steps (mirroring xf_gaussian_diff_accel.cpp):

1. Apply a {FILTER_WIDTH_1}x{FILTER_WIDTH_1} Gaussian blur with sigma={SIGMA1} to img_in
   using zero-padding (XF_BORDER_CONSTANT). This produces G1.

2. Apply a {FILTER_WIDTH_2}x{FILTER_WIDTH_2} Gaussian blur with sigma={SIGMA2} to G1
   using zero-padding (XF_BORDER_CONSTANT). This produces G2.

3. Compute the saturating difference: out = saturate(G1 - G2, 0, 255).
   Uses XF_CONVERT_POLICY_SATURATE: negative results clamp to 0, results above 255
   clamp to 255 (though with uint8 inputs, G1 >= G2 >= 0 so only the floor matters).

Both Gaussian blur stages use the same fixed-point separable convolution from
weightsghcalculation3x3() and xFapplygaussian3x3() in xf_gaussian_filter.hpp.

Weight computation (error-diffusion quantization to 8-bit):
  scale2X = -1 / (sigma^2 * 2)
  cf[i]   = exp(scale2X * (i - 1)^2)  for i in [0, 1, 2]
  sum     = 1 / sum(cf)
  cf[i]  *= sum  (normalize)
  err = 0
  for i in [0, 1, 2]:
      weights[i] = int(cf[i] * 256 + err + 0.5)
      err = cf[i] * 256 - weights[i]

For sigma={SIGMA1}: 1D kernel = [70, 116, 70] (sums to 256).
For sigma={SIGMA2}: 1D kernel = [82, 92, 82] (sums to 256).

The 2D convolution formula (xFapplygaussian3x3, D1..D9 = 3x3 window row-major, D5=center):
  w0, w1 = weights[0], weights[1]
  sum2    = (D4 + D6) * w0 + D5 * w1
  sv0     = D1 + D3 + D7 + D9
  sv1     = D2 + D8
  value1  = sv0 * w0 + sv1 * w1
  total   = value1 * w0 + sum2 * w1
  val     = (total + 32768) >> 16

Border: zero-padding. Both stages operate on uint8 data.
"""

inputs_info = [
    {
        "name": "img_in",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

# Fixed-point 1D Gaussian kernel weights computed by weightsghcalculation3x3()
# from xf_gaussian_filter.hpp using error diffusion quantization.
# sigma=1.0 -> [70, 116, 70] (sums to 256)
# sigma=2.0 -> [82, 92, 82]  (sums to 256)
_WEIGHTS_SIGMA1 = [70, 116, 70]
_WEIGHTS_SIGMA2 = [82, 92, 82]


def _gaussian_fixed_point_3x3(img, weights):
    """Apply a 3x3 fixed-point Gaussian blur matching Vitis xf::cv::GaussianBlur.

    Uses separable integer convolution with the exact formula from
    xFapplygaussian3x3() in xf_gaussian_filter.hpp:
      total = (corners*w0 + top_bot_mids*w1) * w0 + (left_right*w0 + center*w1) * w1
      val   = (total + 32768) >> 16

    Border: zero-padding. Output: uint8 (result is always in [0, 255]).
    """
    H, W = img.shape
    w0, w1 = weights[0], weights[1]

    # Zero-pad by 1 on each side
    padded = [[0] * (W + 2) for _ in range(H + 2)]
    for r in range(H):
        for c in range(W):
            padded[r + 1][c + 1] = int(img[r, c])

    out = np.zeros((H, W), dtype=np.uint8)
    for r in range(H):
        for c in range(W):
            pr, pc = r + 1, c + 1
            D1 = padded[pr - 1][pc - 1]
            D2 = padded[pr - 1][pc]
            D3 = padded[pr - 1][pc + 1]
            D4 = padded[pr][pc - 1]
            D5 = padded[pr][pc]
            D6 = padded[pr][pc + 1]
            D7 = padded[pr + 1][pc - 1]
            D8 = padded[pr + 1][pc]
            D9 = padded[pr + 1][pc + 1]

            sum2   = (D4 + D6) * w0 + D5 * w1
            sv0    = D1 + D3 + D7 + D9
            sv1    = D2 + D8
            value1 = sv0 * w0 + sv1 * w1
            total  = value1 * w0 + sum2 * w1
            val    = (total + 32768) >> 16
            out[r, c] = val
    return out


def behavioral(img_in):
    img = img_in.reshape((HEIGHT, WIDTH))

    # Stage 1: Gaussian blur with sigma=SIGMA1
    g1 = _gaussian_fixed_point_3x3(img, _WEIGHTS_SIGMA1)

    # Stage 2: Gaussian blur of G1 with sigma=SIGMA2
    g2 = _gaussian_fixed_point_3x3(g1, _WEIGHTS_SIGMA2)

    # Saturating subtraction: G1 - G2, clamped to [0, 255] (XF_CONVERT_POLICY_SATURATE)
    diff = g1.astype(np.int32) - g2.astype(np.int32)
    out = np.clip(diff, 0, 255).astype(np.uint8)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
