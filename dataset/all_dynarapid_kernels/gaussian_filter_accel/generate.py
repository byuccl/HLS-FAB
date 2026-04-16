import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
FILTER_WIDTH = 3

# Scalar parameter (compile-time configuration for this generate.py)
SIGMA = 1.0

function_signature = (
    "void gaussian_filter_accel("
    "int32_t* img_inp, "
    "int32_t* img_out)"
)

description = f"""
This HLS kernel applies a Gaussian blur filter to a grayscale image using fixed-point
arithmetic that exactly matches the Vitis xf::cv::GaussianBlur implementation.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for
             the output image (same dimensions as input).

The Gaussian filter is implemented as a separable {FILTER_WIDTH}x{FILTER_WIDTH} convolution.
The 1D kernel weights are computed in weightsghcalculation3x3() from xf_gaussian_filter.hpp:

  scale2X = -1 / (sigma^2 * 2)
  cf[i]   = exp(scale2X * (i - 1)^2)  for i in [0, 1, 2]
  sum     = 1 / sum(cf)
  cf[i]  *= sum  (normalize)
  err = 0
  for i in [0, 1, 2]:
      weights[i] = int(cf[i] * 256 + err + 0.5)   # round with error diffusion
      err = cf[i] * 256 - weights[i]

For sigma = {SIGMA}, this yields the integer 1D kernel: [70, 116, 70] (sums to 256).

The 2D convolution is computed as the outer product of this 1D kernel with itself, using
the optimized form from xFapplygaussian3x3() in xf_gaussian_filter.hpp:

  w0, w1 = weights[0], weights[1]   # weights[2] == weights[0] (symmetric)
  sum2    = (D4 + D6) * w0 + D5 * w1                   # middle row contribution
  sv0     = D1 + D3 + D7 + D9                          # sum of four corners
  sv1     = D2 + D8                                    # sum of top-mid and bot-mid
  value1  = sv0 * w0 + sv1 * w1                        # top+bot rows contribution
  total   = value1 * w0 + sum2 * w1                    # separable combine
  val     = (total + 32768) >> 16                      # round and right-shift by 16

where D1..D9 are the 3x3 window pixels in row-major order (D5 = center).

Border handling: zero-padding (XF_BORDER_CONSTANT). Pixels outside the image boundary
are treated as zero.

Output: truncated to uint8. Since the kernel sums to 256^2 = 65536 and input pixels
are in [0, 255], the output is always in [0, 255].
"""

inputs_info = [
    {
        "name": "img_inp",
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

# Fixed-point 1D Gaussian kernel weights for sigma=1.0, computed by
# weightsghcalculation3x3() from xf_gaussian_filter.hpp with error diffusion.
# Result: [70, 116, 70] (sums to 256).
_WEIGHTS = [70, 116, 70]

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


def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH))
    out = _gaussian_fixed_point_3x3(img, _WEIGHTS)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
