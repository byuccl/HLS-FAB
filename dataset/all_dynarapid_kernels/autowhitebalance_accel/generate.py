import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
CHANNELS = 3
THRESH = 0.9

function_signature = (
    "void autowhitebalance_accel("
    "int32_t* img_inp, "
    "int32_t* img_out)"
)

description = f"""This HLS kernel performs automatic white balance using the Gray World algorithm
on a {HEIGHT}x{WIDTH} 3-channel 8-bit BGR image.

Parameters:
- img_inp: Flat array of {HEIGHT * WIDTH * CHANNELS} uint8 values representing a {HEIGHT}x{WIDTH} BGR image
  in row-major, channel-interleaved order (B0, G0, R0, B1, G1, R1, ...).
- img_out: Flat array of {HEIGHT * WIDTH * CHANNELS} uint8 values for the output white-balanced BGR image.

Algorithm (Gray World AWB):
1. For each pixel, compute minRGB = min(B,G,R) and maxRGB = max(B,G,R).
2. If (maxRGB - minRGB) * 255 > thresh255 * maxRGB, skip the pixel (not gray enough).
   thresh255 = round({THRESH} * 255) = {round(THRESH * 255)}.
3. For qualifying pixels, accumulate per-channel sums: sumB, sumG, sumR.
4. Compute gains: gainB = max_sum / sumB, gainG = max_sum / sumG, gainR = max_sum / sumR,
   where max_sum = max(sumB, sumG, sumR).
5. Normalize gains by dividing each by the maximum gain.
6. Convert gains to fixed-point: i_gain[ch] = round(gain[ch] * 256).
7. Apply gains to each pixel: output[ch] = min(255, (input[ch] * i_gain[ch]) >> 8).
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH * CHANNELS,),
        "total_elements": HEIGHT * WIDTH * CHANNELS,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH * CHANNELS,),
        "total_elements": HEIGHT * WIDTH * CHANNELS,
        "dtype": "int32",
        "iotype": "out",
    },
]


def behavioral(img_inp):
    """Apply Gray World automatic white balance to a BGR image."""
    img = img_inp.reshape((HEIGHT, WIDTH, CHANNELS)).astype(np.int32)
    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    maxval = 255
    thresh255 = int(round(THRESH * 255))

    # Compute channel sums, filtering by grayness threshold
    min_rgb = np.minimum(b, np.minimum(g, r))
    max_rgb = np.maximum(b, np.maximum(g, r))

    # Gray enough condition: (maxRGB - minRGB) * 255 <= thresh255 * maxRGB
    qualify = ((max_rgb - min_rgb) * maxval) <= (thresh255 * max_rgb)

    sumB = int(np.sum(b[qualify]))
    sumG = int(np.sum(g[qualify]))
    sumR = int(np.sum(r[qualify]))

    max_sum = max(sumB, sumG, sumR)
    eps = 0.1

    # Compute gains
    if sumB < eps:
        gainB = 0.0
    else:
        gainB = float(max_sum) / float(sumB)
    if sumG < eps:
        gainG = 0.0
    else:
        gainG = float(max_sum) / float(sumG)
    if sumR < eps:
        gainR = 0.0
    else:
        gainR = float(max_sum) / float(sumR)

    # Normalize by max gain
    gain_max = max(gainB, gainG, gainR)
    if gain_max > 0:
        gainB /= gain_max
        gainG /= gain_max
        gainR /= gain_max

    # Convert to fixed-point (multiply by 256, round)
    i_gainB = int(round(gainB * 256))
    i_gainG = int(round(gainG * 256))
    i_gainR = int(round(gainR * 256))

    # Apply gains
    out = np.zeros_like(img)
    out[:, :, 0] = np.minimum(255, (b * i_gainB) >> 8)
    out[:, :, 1] = np.minimum(255, (g * i_gainG) >> 8)
    out[:, :, 2] = np.minimum(255, (r * i_gainR) >> 8)

    return out.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
