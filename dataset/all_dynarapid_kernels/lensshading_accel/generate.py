import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 676
WIDTH = 1024

function_signature = "void lensshading_accel(int32_t* img_in, int32_t* img_out)"

description = f"""Lens shading correction using distance-based gain.

For a {HEIGHT}x{WIDTH} single-channel 8-bit grayscale image, computes a per-pixel gain based
on the distance from the image center and applies it to correct lens vignetting.

Algorithm:
1. Compute center as (cols >> 1, rows >> 1) = ({WIDTH // 2}, {HEIGHT // 2}).
2. Compute max_distance = sqrt((rows - center_y)^2 + (cols - center_x)^2).
3. For each pixel at position (row, col):
   a. Compute y_dist = center_y - row, x_dist = center_x - col.
   b. Compute normalized distance = sqrt(y_dist^2 + x_dist^2) / max_distance.
   c. Compute gain = 0.01759 * (distance + 28.37)^2 - 13.36.
   d. Compute output = int(pixel_value * gain), where int() truncates toward zero.
   e. Saturate the result to [0, 255].

Parameters:
- img_in: input image, {HEIGHT}x{WIDTH} pixels, 8-bit unsigned, row-major flattened.
- img_out: output image, same dimensions and type.
"""

inputs_info = [
    {
        "name": "img_in",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "int32",
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "iotype": "out",
        "array_dimension": 1,
        "dtype": "int32",
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "min": 0,
        "max": 255,
    },
]


def behavioral(img_in):
    img = img_in.reshape(HEIGHT, WIDTH).astype(np.float32)

    center_x = WIDTH >> 1    # 512
    center_y = HEIGHT >> 1   # 338

    y_distance = HEIGHT - center_y
    x_distance = WIDTH - center_x

    max_distance = np.sqrt(np.float32(y_distance * y_distance) + np.float32(x_distance * x_distance))

    rows_idx = np.arange(HEIGHT, dtype=np.float32)
    cols_idx = np.arange(WIDTH, dtype=np.float32)

    y_dist = np.float32(center_y) - rows_idx
    x_dist = np.float32(center_x) - cols_idx

    y_dist_2 = y_dist * y_dist
    x_dist_2 = x_dist * x_dist

    # Compute distance for each pixel
    dist = np.sqrt(y_dist_2[:, None] + x_dist_2[None, :])
    normalized_dist = dist / max_distance

    a = np.float32(0.01759)
    b = np.float32(28.37)
    c = np.float32(13.36)

    gain = a * (normalized_dist + b) * (normalized_dist + b) - c

    result = img * gain

    # int() truncation toward zero
    result = np.trunc(result).astype(np.int32)

    # Saturate to [0, 255]
    result = np.clip(result, 0, 255)

    return result.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
