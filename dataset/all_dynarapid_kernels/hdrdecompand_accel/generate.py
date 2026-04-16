import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 1080
WIDTH = 1920

function_signature = "void hdrdecompand_accel(int32_t* in_ptr, int32_t* out_ptr)"

description = f"""HDR decompanding kernel that applies a piecewise linear transfer function to each pixel
of a Bayer-pattern image. The image is {WIDTH}x{HEIGHT} single-channel 16-bit unsigned input,
producing a 32-bit signed output.

Each pixel is classified into one of 3 color channels (R=0, G=1, B=2) based on its position
in a Bayer BG pattern. For BG pattern, even-row/even-col is R(0), even-row/odd-col is G(1),
odd-row/even-col is G(1), odd-row/odd-col is B(2).

For each color channel, a 4-segment piecewise linear function is applied:
  - Segment 0: if pixel < 8192, output = 4 * (pixel - 0)
  - Segment 1: if pixel < 22528, output = 16 * (pixel - 6144)
  - Segment 2: if pixel < 34816, output = 64 * (pixel - 18432)
  - Segment 3: otherwise, output = 512 * (pixel - 32768)

The same piecewise linear parameters are used for all 3 color channels.
The output is a flat array of 32-bit signed integers.
"""

inputs_info = [
    {
        "name": "in_ptr",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "dtype": "int32",
        "total_elements": HEIGHT * WIDTH,
        "iotype": "in",
        "min": 0,
        "max": 65535,
    },
    {
        "name": "out_ptr",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "dtype": "int32",
        "total_elements": HEIGHT * WIDTH,
        "iotype": "out",
    },
]

# Piecewise linear params for 16-bit to 24-bit decompanding
# params[color_idx][segment][0=threshold, 1=slope, 2=offset]
# Same params for all 3 color channels
PARAMS = np.array([
    [[8192, 4, 0], [22528, 16, 6144], [34816, 64, 18432], [65536, 512, 32768]],
    [[8192, 4, 0], [22528, 16, 6144], [34816, 64, 18432], [65536, 512, 32768]],
    [[8192, 4, 0], [22528, 16, 6144], [34816, 64, 18432], [65536, 512, 32768]],
], dtype=np.int64)

# BG bayer pattern: row_incr=0, col_incr=0
BAYERP = 0
ROW_INCR = 0
COL_INCR = 0


def behavioral(in_ptr):
    img = in_ptr.reshape(HEIGHT, WIDTH).astype(np.int64)
    out = np.zeros((HEIGHT, WIDTH), dtype=np.int64)

    row_idx = np.arange(HEIGHT).reshape(-1, 1) + ROW_INCR
    col_idx = np.arange(WIDTH).reshape(1, -1) + COL_INCR
    row_rem = row_idx & 1
    col_rem = col_idx & 1
    color_idx = row_rem + col_rem  # 0, 1, or 2

    for cidx in range(3):
        mask_c = (color_idx == cidx)
        pxl = img.copy()

        # Segment 0: pxl < threshold[0]
        seg0 = pxl < PARAMS[cidx, 0, 0]
        # Segment 1: threshold[0] <= pxl < threshold[1]
        seg1 = (pxl >= PARAMS[cidx, 0, 0]) & (pxl < PARAMS[cidx, 1, 0])
        # Segment 2: threshold[1] <= pxl < threshold[2]
        seg2 = (pxl >= PARAMS[cidx, 1, 0]) & (pxl < PARAMS[cidx, 2, 0])
        # Segment 3: pxl >= threshold[2]
        seg3 = pxl >= PARAMS[cidx, 2, 0]

        result = np.zeros_like(pxl)
        result[seg0] = PARAMS[cidx, 0, 1] * (pxl[seg0] - PARAMS[cidx, 0, 2])
        result[seg1] = PARAMS[cidx, 1, 1] * (pxl[seg1] - PARAMS[cidx, 1, 2])
        result[seg2] = PARAMS[cidx, 2, 1] * (pxl[seg2] - PARAMS[cidx, 2, 2])
        result[seg3] = PARAMS[cidx, 3, 1] * (pxl[seg3] - PARAMS[cidx, 3, 2])

        out[mask_c] = result[mask_c]

    return out.astype(np.int32).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
