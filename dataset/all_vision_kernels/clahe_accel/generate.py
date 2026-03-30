import numpy as np
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

# Note: In xf_clahe_accel.cpp (line 86), clahe_accel_i is called with tilesX and tilesY SWAPPED:
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
# mTileWidthNPCAligned: for NPC=1, XF_BITSHIFT=0, so it equals tileW unchanged
_tileWNPC_val = (_tileW_val >> 0) << 0  # = _tileW_val

description = (
    "This HLS kernel applies Contrast Limited Adaptive Histogram Equalization (CLAHE)\n"
    "to a grayscale image using exact integer arithmetic.\n"
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
    "It uses exact integer arithmetic matching the Vitis HLS C++ implementation.\n"
    "\n"
    "Tile dimension computation:\n"
    f"  tileH = HEIGHT // tilesY = {HEIGHT // TILES_Y}\n"
    f"  If tileH * tilesY < HEIGHT: tileH += 1  -> tileH = {_tileH_val}\n"
    f"  tileW = WIDTH // tilesX = {WIDTH // TILES_X}\n"
    f"  If tileW * tilesX < WIDTH: tileW += 1   -> tileW = {_tileW_val}\n"
    "  tileSize = tileH * tileW\n"
    "  clipValue = max((clip * tileH * tileW) >> 8, 1)  [>> 8 = XF_DTPIXELDEPTH for XF_8UC1]\n"
    "\n"
    "Double-buffer scheme:\n"
    "  The kernel uses static LUT arrays and a flag to alternate between two LUTs.\n"
    "  On the first call, LUT2 (all zeros) is used for interpolation (warmup output discarded).\n"
    "  On the second call, LUT1 (computed from the first call's image) is used for interpolation.\n"
    "  behavioral() simulates both passes: pass 1 builds the LUT, pass 2 interpolates using it.\n"
    "\n"
    "Per-tile histogram building (populateLutBlk):\n"
    "  NPC=1, so each pixel is processed individually.\n"
    "  Pixels are read in pairs (j, j+1); pixel at even NPC-aligned column goes to\n"
    "  sub-lut[0], pixel at odd NPC-aligned column goes to sub-lut[1].\n"
    "  Each sub-lut bin is capped at clipValue during accumulation;\n"
    "  excess counts go to clipCounter for that tile.\n"
    "  counterX advances by 2 per pixel-pair; histX increments when counterX >= tileW.\n"
    "  counterY increments every row; histY increments every tileH rows.\n"
    "\n"
    "Histogram clipping and CDF normalization (clipLut):\n"
    "  For each tile (i, j):\n"
    "    add = clipCounter[i][j]\n"
    "    For each bin k (0..255): v = sub-lut[0][k] + sub-lut[1][k]\n"
    "      if v > clipValue: add += (v - clipValue)\n"
    "    div = add >> 8,  rem = add & 0xFF\n"
    "    cumsum = 0\n"
    "    For each bin k (0..255):\n"
    "      v = sub-lut[0][k] + sub-lut[1][k]\n"
    "      if v > clipValue: v = clipValue\n"
    "      v += div\n"
    "      if rem > 0:\n"
    "        step = max(256 // rem, 1)\n"
    "        if k % step == 0 and k < rem * step: v++\n"
    "      cumsum += v\n"
    "      lut[k] = min(((cumsum * 255) // tileSize), 255)  [using (sum<<8)-sum = sum*255]\n"
    "    (stored in both sub-lut[0] and sub-lut[1])\n"
    "\n"
    "Bilinear interpolation (interpolate):\n"
    "  counterY starts at tileH // 2, histY starts at -1\n"
    "  counterX starts at tileW // 2, histX starts at -1  (reset each row)\n"
    "  For each pixel (row, col):\n"
    "    Y1 = max(histY, 0), Y2 = min(histY+1, tilesY-1)\n"
    "    X1 = max(histX, 0), X2 = min(histX+1, tilesX-1)\n"
    "    a1 = lut[Y1][X1][0][pixel], a2 = lut[Y1][X2][0][pixel]\n"
    "    b1 = lut[Y2][X1][1][pixel], b2 = lut[Y2][X2][1][pixel]\n"
    "    xa1 = tileW - counterX, xa = counterX\n"
    "    ya1 = tileH - counterY, ya = counterY\n"
    "    t1 = a1*xa1 + a2*xa\n"
    "    t2 = b1*xa1 + b2*xa\n"
    "    t3 = t1*ya1 + t2*ya\n"
    "    output = min(t3 // tileSize, 255)\n"
    "    counterX += 1; when >= tileW: counterX=0, histX++\n"
    "    counterY += 1; when == tileH: histY++, counterY=0\n"
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


def _clahe_build_luts(img, tileH, tileW, tilesY, tilesX, clipValue):
    """Build per-tile LUTs exactly matching Vitis populateLutBlk + clipLut.

    Returns lut[tilesY][tilesX][2][256] as numpy array (dtype int32).
    sub-lut[l][k] for l in {0,1} is the histogram for even/odd NPC-aligned columns.
    After clipLut, all sub-luts for a tile contain the same CDF-normalized values.
    """
    # lut[ty][tx][sublut_idx][bin]
    lut = np.zeros((tilesY, tilesX, 2, 256), dtype=np.int32)
    clip_counter = np.zeros((tilesY, tilesX), dtype=np.int64)

    rows, cols = img.shape
    # NPC=1, so mColsNPCAligned = cols >> 0 = cols
    # counterX advances by 2 * (1<<0) = 2 per pair; tile boundary at tileW (NPC-aligned = tileW)

    for row in range(rows):
        histY = row // tileH
        if histY >= tilesY:
            histY = tilesY - 1
        counterX = 0
        histX = 0
        # Process in pairs (j=0,2,4,...) — j is NPC-aligned column index
        # For NPC=1, NPC-aligned col == pixel col
        j = 0
        while j < cols:
            # pxl1 at column j (sub-lut index 0)
            a1 = int(img[row, j])
            val1 = lut[histY, histX, 0, a1] + 1
            if val1 <= clipValue:
                lut[histY, histX, 0, a1] = val1
            else:
                clip_counter[histY, histX] += 1

            # pxl2 at column j+1 (sub-lut index 1), or 0 if beyond image
            if j + 1 < cols:
                a2 = int(img[row, j + 1])
                val2 = lut[histY, histX, 1, a2] + 1
                if val2 <= clipValue:
                    lut[histY, histX, 1, a2] = val2
                else:
                    clip_counter[histY, histX] += 1

            counterX += 2  # advance by 2 (NPC=1, so 2 NPC-aligned cols per pair)
            if counterX >= tileW:
                histX += 1
                counterX = 0

            j += 2

    # clipLut: for each tile, combine sub-luts, apply clipping redistribution, compute CDF
    tile_size = tileH * tileW
    for ty in range(tilesY):
        for tx in range(tilesX):
            add = int(clip_counter[ty, tx])

            # First pass: accumulate excess counts
            for k in range(256):
                v = int(lut[ty, tx, 0, k]) + int(lut[ty, tx, 1, k])
                if v > clipValue:
                    add += v - clipValue

            div = add >> 8
            rem = add & 0xFF

            # Second pass: compute CDF and normalize
            cumsum = 0
            for k in range(256):
                v = int(lut[ty, tx, 0, k]) + int(lut[ty, tx, 1, k])
                if v > clipValue:
                    v = clipValue
                v += div

                if rem > 0:
                    step = 256 // rem
                    if step < 1:
                        step = 1
                    if (k % step == 0) and (k < rem * step):
                        v += 1

                cumsum += v
                # fi = ((sum << 8) - sum) / tileSize = sum * 255 / tileSize (integer)
                fi = ((cumsum << 8) - cumsum) // tile_size
                if fi > 255:
                    fi = 255

                lut[ty, tx, 0, k] = fi
                lut[ty, tx, 1, k] = fi

    return lut


def _clahe_interpolate(img, lut, tileH, tileW, tilesY, tilesX):
    """Apply bilinear interpolation exactly matching Vitis interpolate().

    Uses lut[tilesY][tilesX][2][256].
    sub-lut index 0 is used for a1, a2 (Y1 row);
    sub-lut index 1 is used for b1, b2 (Y2 row).
    """
    rows, cols = img.shape
    out = np.zeros((rows, cols), dtype=np.uint8)
    tile_size = tileH * tileW

    counterY = tileH >> 1
    histY = -1

    for row in range(rows):
        counterX = tileW >> 1
        histX = -1

        for col in range(cols):
            pixel = int(img[row, col])

            Y1 = max(histY, 0)
            Y2 = histY + 1
            if Y2 >= tilesY:
                Y2 = tilesY - 1

            X1 = max(histX, 0)
            X2 = histX + 1
            if X2 >= tilesX:
                X2 = tilesX - 1

            # sub-lut index 0 for a1, a2 (upper tiles)
            # sub-lut index 1 for b1, b2 (lower tiles)
            a1 = int(lut[Y1, X1, 0, pixel])
            a2 = int(lut[Y1, X2, 0, pixel])
            b1 = int(lut[Y2, X1, 1, pixel])
            b2 = int(lut[Y2, X2, 1, pixel])

            xa1 = tileW - counterX
            xa = counterX
            ya1 = tileH - counterY
            ya = counterY

            t1 = a1 * xa1 + a2 * xa
            t2 = b1 * xa1 + b2 * xa
            t3 = t1 * ya1 + t2 * ya
            t4 = t3 // tile_size

            if t4 >= 256:
                out[row, col] = 255
            else:
                out[row, col] = t4

            # Advance counterX; for NPC=1, increment by 1
            counterX += 1
            if counterX >= tileW:
                counterX = 0
                histX += 1

        counterY += 1
        if counterY == tileH:
            histY += 1
            counterY = 0

    return out


def behavioral(img_in):
    img = img_in.reshape((HEIGHT, WIDTH)).astype(np.uint8)

    # Compute tile dimensions
    tileH = HEIGHT // TILES_Y
    if tileH * TILES_Y < HEIGHT:
        tileH += 1
    tileW = WIDTH // TILES_X
    if tileW * TILES_X < WIDTH:
        tileW += 1

    # Clip value: max((clip * tileH * tileW) >> 8, 1)
    clipValue = max((CLIP * tileH * tileW) >> 8, 1)

    # Pass 1: build LUT from the image (simulates first kernel call, which writes _lut1)
    lut = _clahe_build_luts(img, tileH, tileW, TILES_Y, TILES_X, clipValue)

    # Pass 2: interpolate using the LUT just built (simulates second kernel call, which reads _lut1)
    out = _clahe_interpolate(img, lut, tileH, tileW, TILES_Y, TILES_X)

    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
