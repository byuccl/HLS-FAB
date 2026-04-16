import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 128
WIDTH = 128
WINDOW_SIZE = 3
WIN_SZ_SQ = WINDOW_SIZE * WINDOW_SIZE  # 9
# IN_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# OUT_TYPE = XF_8UC1 -> ap_uint<8>, dtype: uint8
# INPUT_PTR_WIDTH = 8, OUTPUT_PTR_WIDTH = 8
# Border: XF_BORDER_REPLICATE

function_signature = (
    "void modefilter_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

description = f"""\
This HLS kernel applies a {WINDOW_SIZE}×{WINDOW_SIZE} mode filter to a grayscale image.
The mode filter outputs the most frequently occurring value in each {WINDOW_SIZE}×{WINDOW_SIZE}
neighbourhood, using a non-standard counting algorithm.

Parameters:
    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing
             a {HEIGHT}×{WIDTH} grayscale image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 output pixel values.

Border handling: replicate (edge pixels are mirrored at the border).

Mode computation for each {WIN_SZ_SQ}-element neighbourhood (indexed 0..{WIN_SZ_SQ-1}):
1. Count phase: for each element i = 0..{WIN_SZ_SQ-1}, compute count[i] = 1 + number of
   elements j in [i+1, {WIN_SZ_SQ-2}] (NOT including index {WIN_SZ_SQ-1}) that equal element i.
   Find max_count = max(count[i]) over all i.
2. Index phase: scan i = 0..{WIN_SZ_SQ-1} again (same j range [i+1, {WIN_SZ_SQ-2}]),
   and set idx = i whenever count[i] == max_count (updating on each tie; last tie wins).
3. Output = neighbourhood[idx].

Note: element {WIN_SZ_SQ-1} (index {WIN_SZ_SQ-1}) is never compared as a 'j' value,
so its count is always 1. If all elements are distinct (max_count=1), idx is set to
the last i where count==1, which is always {WIN_SZ_SQ-1}.
"""

inputs_info = [
    {
        "name": "img_in",
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

def _mode_9(arr):
    """Non-standard mode matching xf_modefilter.hpp for WIN_SZ_SQ=9.
    Inner j loop: j in [i+1, WIN_SZ_SQ-2] = [i+1, 7] (excludes index 8).
    Two passes: first finds max_count, second finds last idx with that count."""
    N = WIN_SZ_SQ  # 9
    # First pass: find max_count
    max_count = 0
    for i in range(N):
        count = 1
        for j in range(i + 1, N - 1):  # j < N-1 = 8
            if arr[i] == arr[j]:
                count += 1
        if count > max_count:
            max_count = count
    # Second pass: find last idx where count == max_count
    idx = 0
    for i in range(N):
        count = 1
        for j in range(i + 1, N - 1):
            if arr[i] == arr[j]:
                count += 1
        if count == max_count:
            idx = i
    return arr[idx]

# Border: replicate (edge pixels repeated outside image boundary)
# Division: none (mode filter)
# Accumulator: count as int (comparison only)
# Output: uint8, direct copy of most-frequent element
def behavioral(img_in):
    img = img_in.reshape((HEIGHT, WIDTH))
    # Replicate border padding
    pad = WINDOW_SIZE // 2  # = 1
    padded = np.pad(img, pad, mode='edge').astype(np.uint8)
    out = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    for r in range(HEIGHT):
        for c in range(WIDTH):
            win = padded[r:r + WINDOW_SIZE, c:c + WINDOW_SIZE].flatten()
            out[r, c] = _mode_9(win)
    return out.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
