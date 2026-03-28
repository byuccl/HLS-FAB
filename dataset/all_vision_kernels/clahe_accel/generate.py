import numpy as np
import cv2
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 540
WIDTH = 960
# IN_TYPE = XF_8UC1, NPC = XF_NPPC1 (NPC=1)
# INPUT_PTR_WIDTH = 512

# Runtime parameters (hardcoded)
CLIP = 40
TILES_Y = 4   # TILES_Y_MAX from config (runtime tilesY must be <= TILES_Y_MAX=4)
TILES_X = 4   # TILES_X_MAX from config

# Note: In xf_clahe_accel.cpp, clahe_accel_i is called with tilesX and tilesY SWAPPED:
#   clahe_accel_i(..., clip, tilesX, tilesY)
# The internal CLAHETile sees (height, width, clip, tilesX_passed_as_tilesY, tilesY_passed_as_tilesX).
# Since TILES_Y==TILES_X==4, the swap has no effect here.

function_signature = (
    "void clahe_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out)"
)

_tileH_val = (HEIGHT // TILES_Y) + (1 if (HEIGHT // TILES_Y) * TILES_Y < HEIGHT else 0)
_tileW_val = (WIDTH // TILES_X) + (1 if (WIDTH // TILES_X) * TILES_X < WIDTH else 0)

description = (
    "This HLS kernel applies Contrast Limited Adaptive Histogram Equalization (CLAHE)\n"
    "to a grayscale image.\n"
    "\n"
    "Parameters:\n"
    f"    img_in:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values representing\n"
    f"             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.\n"
    f"    img_out: Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel values for\n"
    "             the CLAHE-processed output image (same dimensions as input).\n"
    "\n"
    "Algorithm parameters (hardcoded):\n"
    f"    clip   = {CLIP}   (clip limit parameter)\n"
    f"    tilesY = {TILES_Y}   (number of tiles along Y axis)\n"
    f"    tilesX = {TILES_X}   (number of tiles along X axis)\n"
    "\n"
    "The kernel implements CLAHE with bilinear interpolation between tile LUTs.\n"
    "\n"
    "Tile dimension computation:\n"
    f"  tileH = HEIGHT // tilesY = {HEIGHT // TILES_Y}\n"
    f"  If tileH * tilesY < HEIGHT: tileH += 1  -> tileH = {_tileH_val}\n"
    f"  tileW = WIDTH // tilesX = {WIDTH // TILES_X}\n"
    f"  If tileW * tilesX < WIDTH: tileW += 1   -> tileW = {_tileW_val}\n"
    "  tileSize = tileH * tileW\n"
    "  clipValue = max((clip * tileH * tileW) >> 8, 1)\n"
    "\n"
    "Per-tile histogram building (populateLutBlk):\n"
    "  Pixels are processed in pairs (j, j+1). Pixel at even column index goes to\n"
    "  sub-lut[0], pixel at odd column index goes to sub-lut[1]. Each sub-lut bin\n"
    "  is capped at clipValue during accumulation; excess counts go to clipCounter.\n"
    "  Tile boundaries advance counterX by 2 per pixel-pair; histX increments when\n"
    "  counterX >= tileW. histY increments every tileH rows.\n"
    "\n"
    "Histogram clipping and CDF normalization (clipLut):\n"
    "  combined[k] = sub-lut[0][k] + sub-lut[1][k]\n"
    "  add = clipCounter + sum_k max(0, combined[k] - clipValue)  (total excess)\n"
    "  div = add >> 8,  rem = add & 0xFF\n"
    "  For each bin k (in order):\n"
    "    v = min(combined[k], clipValue) + div\n"
    "    if rem > 0: step = max(256 // rem, 1); if k % step == 0 and k < rem*step: v++\n"
    "    cumsum += v\n"
    "    lut[k] = min((cumsum * 255) // tileSize, 255)\n"
    "  (lut[k] is stored in both sub-lut[0] and sub-lut[1])\n"
    "\n"
    "Bilinear interpolation (interpolate):\n"
    "  The kernel uses a double-buffer scheme across calls. The second call applies\n"
    "  the LUT computed during the first call (from the same input) via bilinear\n"
    "  interpolation in tile space.\n"
    "  counterY starts at tileH // 2, histY starts at -1\n"
    "  counterX starts at tileW // 2, histX starts at -1\n"
    "  For each pixel (row, col):\n"
    "    Y1 = max(histY, 0), Y2 = min(histY+1, tilesY-1)\n"
    "    X1 = max(histX, 0), X2 = min(histX+1, tilesX-1)\n"
    "    a1 = lut[Y1][X1][pixel], a2 = lut[Y1][X2][pixel]\n"
    "    b1 = lut[Y2][X1][pixel], b2 = lut[Y2][X2][pixel]\n"
    "    xa1 = tileW - counterX, xa = counterX\n"
    "    ya1 = tileH - counterY, ya = counterY\n"
    "    output = min((a1*xa1 + a2*xa)*ya1 + (b1*xa1 + b2*xa)*ya) // tileSize, 255)\n"
    "    counterX increments by 1; when >= tileW: counterX=0, histX++\n"
    "    counterY increments by 1; when == tileH: histY++, counterY=0\n"
)

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


def behavioral(img_in):
    img = img_in.reshape((540, 960)).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(4, 4))
    out = clahe.apply(img)
    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
