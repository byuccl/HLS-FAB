import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128

function_signature = "void badpixelcorrection_accel(ap_uint<8>* img_inp, ap_uint<8>* img_out)"

description = f"""Bad pixel correction on a {HEIGHT}x{WIDTH} single-channel 8-bit grayscale image.

For each pixel, a 5x5 neighborhood is examined. From the 5x5 patch centered on the current pixel, 9 pixels are sampled at every-other position (step of 2 in both row and column), forming a 3x3 sub-grid. The center of this sub-grid is the current pixel. The 8 surrounding samples form the neighbor set.

The minimum and maximum of the 8 neighbors are computed. If the center pixel is below the minimum, the output is the minimum. If the center pixel is above the maximum, the output is the maximum. Otherwise, the center pixel is passed through unchanged.

At image borders, out-of-bounds pixels are treated as zero (constant border padding).

Parameters:
- img_inp: flattened {HEIGHT}x{WIDTH} input image (uint8)
- img_out: flattened {HEIGHT}x{WIDTH} output image (uint8)
"""

inputs_info = [
    {
        "name": "img_inp",
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


def behavioral(img_inp):
    img = img_inp.reshape(HEIGHT, WIDTH).astype(np.int32)
    # Pad with zeros (constant border), 2 pixels on each side for 5x5 step-2 kernel
    padded = np.pad(img, 2, mode='constant', constant_values=0)
    out = np.empty((HEIGHT, WIDTH), dtype=np.int32)

    for r in range(HEIGHT):
        for c in range(WIDTH):
            # Extract 5x5 patch from padded image (center at padded[r+2][c+2])
            # Sample at step 2: rows r, r+2, r+4 and cols c, c+2, c+4
            array = []
            for kr in range(0, 5, 2):
                for kc in range(0, 5, 2):
                    array.append(padded[r + kr, c + kc])
            # array[0..8]: 9 sampled pixels, array[4] is center
            # neighbors = array[0..3] + array[5..8]
            neighbors = array[:4] + array[5:]
            min_val = min(neighbors)
            max_val = max(neighbors)
            center = array[4]
            if center < min_val:
                out[r, c] = min_val
            elif center > max_val:
                out[r, c] = max_val
            else:
                out[r, c] = center

    return out.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
