import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
MAX_BOXES = 5
NUM_BOX = 3
CHANNELS = 4

function_signature = (
    "void boundingbox_accel("
    "int32_t* in_img, "
    "int32_t* roi, "
    "int32_t* color_info)"
)

description = f"""This HLS kernel draws bounding box rectangles on a {HEIGHT}x{WIDTH} 4-channel 16-bit unsigned image.

The image has {CHANNELS} channels with 16 bits per channel, packed into 64-bit words (one pixel per word).
The image is modified in-place.

Parameters:
- in_img: Input/output image of {HEIGHT*WIDTH} 64-bit words. Each word packs 4 channels of 16-bit unsigned values
  (channel 0 in bits [15:0], channel 1 in bits [31:16], channel 2 in bits [47:32], channel 3 in bits [63:48]).
- roi: Array of {MAX_BOXES*4} 32-bit integers. For each box i (0 to {NUM_BOX-1}):
  roi[i*4+0]=x, roi[i*4+1]=y, roi[i*4+2]=box_height, roi[i*4+3]=box_width.
  {NUM_BOX} boxes are drawn. Remaining entries are unused.
- color_info: Array of {MAX_BOXES*CHANNELS} 32-bit integers. For each box i,
  color_info[i*{CHANNELS}+j] gives the color value for channel j (only lower 8 bits used, value 0-255).

For each bounding box, the kernel draws a 1-pixel-thick rectangle border:
- Top edge: row y, columns x to x+box_width-1
- Bottom edge: row y+box_height-1, columns x to x+box_width-1
- Left edge: rows y+1 to y+box_height-2, column x
- Right edge: rows y+1 to y+box_height-2, column x+box_width-1

Each border pixel is set to the box's color packed into a 64-bit word (16 bits per channel).
"""

inputs_info = [
    {
        "name": "roi",
        "array_dimension": 1,
        "shape": (MAX_BOXES * 4,),
        "total_elements": MAX_BOXES * 4,
        "dtype": "int32",
        "iotype": "in",
        "min": 2,
        "max": 30,
    },
    {
        "name": "color_info",
        "array_dimension": 1,
        "shape": (MAX_BOXES * CHANNELS,),
        "total_elements": MAX_BOXES * CHANNELS,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "in_img",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "inout",
        "min": 0,
        "max": 2**64 - 1,
    },
]


def behavioral(roi, color_info, in_img):
    img = in_img.copy().astype(np.uint64)
    for b in range(3):
        x = max(0, int(roi[b * 4]))
        y = max(0, int(roi[b * 4 + 1]))
        bh = int(roi[b * 4 + 2])
        bw = int(roi[b * 4 + 3])
        bw = min(128 - x, bw)
        bh = min(128 - y, bh)
        if bw < 2 or bh < 2:
            continue
        color_val = np.uint64(0)
        for ch in range(4):
            c = np.uint64(color_info[b * 4 + ch] & 0xFF)
            color_val |= c << np.uint64(ch * 16)
        c_start, c_end = x, x + bw
        r_start, r_end = y, y + bh
        for j in range(c_start, c_end):
            img[r_start * 128 + j] = color_val
            img[(r_end - 1) * 128 + j] = color_val
        for i in range(r_start + 1, r_end - 1):
            img[i * 128 + c_start] = color_val
            img[i * 128 + (c_end - 1)] = color_val
    return img.astype(np.uint64).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
