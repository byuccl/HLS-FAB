import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 64
WIDTH = 64

# IN_TYPE = XF_8UC1 -> 8-bit unsigned grayscale
# OUT_TYPE = XF_32FC1 -> 32-bit float (IEEE 754), stored as ap_uint<32>
# INPUT_PTR_WIDTH = 64 (packing), OUTPUT_PTR_WIDTH = 32
# NPPCX = XF_NPPC1 -> one pixel per clock
# KMED = 25 -> WINDOW_SIZE for the sliding window
# XF_USE_URAM = false

WINDOW_SIZE = 25

# The top-level function signature uses the pixel bit-width for
# ap_uint width:
#   - img_curr, img_prev: ap_uint<8>* (IN_TYPE = XF_8UC1 -> 8 bits)
#   - img_outx, img_outy: ap_uint<32>* (OUT_TYPE = XF_32FC1 -> 32 bits)
# rows and cols are scalar control params, excluded from signature.

function_signature = (
    "void dense_non_pyr_of_accel("
    "ap_uint<8>* img_curr, "
    "ap_uint<8>* img_prev, "
    "ap_uint<32>* img_outx, "
    "ap_uint<32>* img_outy)"
)

description = f"""
This HLS kernel computes dense non-pyramidal Lucas-Kanade optical flow
between two consecutive grayscale frames.

Parameters:
    img_curr:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel
               values for the current grayscale image ({HEIGHT}x{WIDTH},
               row-major).
    img_prev:  Pointer to a flat array of {HEIGHT * WIDTH} uint8 pixel
               values for the previous grayscale image ({HEIGHT}x{WIDTH},
               row-major).
    img_outx:  Pointer to a flat array of {HEIGHT * WIDTH} uint32 values
               representing the horizontal (x) component of optical flow.
               Each value is an IEEE 754 float32 reinterpreted as uint32.
    img_outy:  Pointer to a flat array of {HEIGHT * WIDTH} uint32 values
               representing the vertical (y) component of optical flow.
               Each value is an IEEE 754 float32 reinterpreted as uint32.

Algorithm (Dense Non-Pyramidal LK Optical Flow, WINDOW_SIZE={WINDOW_SIZE}):
  1. Read both frames into a line buffer of depth WINDOW_SIZE+1={WINDOW_SIZE + 1}
     rows.
  2. At each pixel position (r, c), compute spatial gradients Ix, Iy and
     temporal gradient It using the line buffer contents:
       - Ix = (frame0[row_top, col] - frame0[row_top, col_prev]) / 2
         (central-difference-like using the line buffer window).
       - Iy = (frame0[row_top+1, col_prev] - frame0[row_top-1, col_prev]) / 2
       - It = frame0[row_mid, col_prev] - frame1[row_mid, col_prev]
     where row_top and row_bot correspond to the top (index 1) and bottom
     (index WINDOW_SIZE-1) of the sliding column window.
  3. Maintain running column sums and row sums (sliding window of size
     WINDOW_SIZE) for the products: Ix*Ix, Ix*Iy, Iy*Iy, It*Ix, It*Iy.
  4. Solve the 2x2 linear system at each pixel:
       det = sum(Ix^2)*sum(Iy^2) - sum(Ix*Iy)^2
       if det > 1.0 and r >= WINDOW_SIZE and c >= WINDOW_SIZE+1:
         fx = (sum(Iy^2)*sum(It*Ix) - sum(Ix*Iy)*sum(It*Iy)) / det
         fy = (-sum(Ix*Iy)*sum(It*Ix) + sum(Ix^2)*sum(It*Iy)) / det
       else:
         fx = fy = 0.0
  5. Output float32 flow values reinterpreted as uint32.
"""

inputs_info = [
    {
        "name": "img_curr",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_prev",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint8",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_outx",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint32",
        "iotype": "out",
    },
    {
        "name": "img_outy",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "uint32",
        "iotype": "out",
    },
]


def behavioral(img_curr, img_prev):
    """
    Cycle-accurate model of the dense non-pyramidal LK optical flow HW
    (NPPC1 path: fpga_optflow8 -> flowWrap -> lbWrapper + computeSums +
    computeFlow + writeOutput8).

    The HW pipeline processes pixels in raster order. A line buffer of
    WINDOW_SIZE+1 rows feeds column vectors into computeSums, which
    maintains sliding-window column sums and row sums of gradient
    products. computeFlow solves the 2x2 system per pixel.
    """
    rows = HEIGHT
    cols = WIDTH
    WS = WINDOW_SIZE
    LB_DEPTH = WS + 1  # 26

    frame0 = img_curr.reshape((rows, cols)).astype(np.int32)
    frame1 = img_prev.reshape((rows, cols)).astype(np.int32)

    # ---- Simulate lbWrapper ----
    # The line buffer stores the last LB_DEPTH rows of input.
    # At each (r, c), lb[i][c] for i in 0..LB_DEPTH-1 gives the
    # column vector that is output as img1Col / img2Col.
    # After processing pixel (r, c):
    #   lb[LB_DEPTH-1][c] = frame[r][c] (newly read pixel)
    #   lb[i][c] = lb[i+1][c] (shifted up) for i < LB_DEPTH-1
    # So at (r, c), after the shift and new-pixel insertion:
    #   lb[LB_DEPTH-1][c] = frame0[r, c]
    #   lb[LB_DEPTH-2][c] = frame0[r-1, c] (if r>=1)
    #   ...
    #   lb[k][c] = frame0[r - (LB_DEPTH-1-k), c] if that row >= 0, else 0
    # The column vector output is the state AFTER the shift:
    #   img1Col[i] = lb[i][c] after shift = frame0[r - (LB_DEPTH-1-i), c]

    # ---- Simulate computeSums ----
    # State variables (matching HW static vars)
    img1Win = np.zeros(2 * LB_DEPTH, dtype=np.int32)
    img2Win = np.zeros(LB_DEPTH, dtype=np.int32)

    csIxix = np.zeros(cols, dtype=np.int64)
    csIxiy = np.zeros(cols, dtype=np.int64)
    csIyiy = np.zeros(cols, dtype=np.int64)
    csDix = np.zeros(cols, dtype=np.int64)
    csDiy = np.zeros(cols, dtype=np.int64)

    cbIxix = np.zeros(cols, dtype=np.int64)
    cbIxiy = np.zeros(cols, dtype=np.int64)
    cbIyiy = np.zeros(cols, dtype=np.int64)
    cbDix = np.zeros(cols, dtype=np.int64)
    cbDiy = np.zeros(cols, dtype=np.int64)

    ixix_acc = np.int64(0)
    ixiy_acc = np.int64(0)
    iyiy_acc = np.int64(0)
    dix_acc = np.int64(0)
    diy_acc = np.int64(0)

    # Line buffer state (all zeros initially)
    lb1 = np.zeros((LB_DEPTH, cols), dtype=np.int32)
    lb2 = np.zeros((LB_DEPTH, cols), dtype=np.int32)

    fx_out = np.zeros((rows, cols), dtype=np.float32)
    fy_out = np.zeros((rows, cols), dtype=np.float32)

    zIdx_init = -(WS - 2)
    nIdx_init = zIdx_init + WS - 2  # = 0

    for r in range(rows):
        zIdx = zIdx_init
        nIdx = nIdx_init
        ixix_acc = np.int64(0)
        ixiy_acc = np.int64(0)
        iyiy_acc = np.int64(0)
        dix_acc = np.int64(0)
        diy_acc = np.int64(0)

        for c in range(cols):
            # --- lbWrapper: shift up, read new pixel ---
            img1Col_ = np.zeros(LB_DEPTH, dtype=np.int32)
            img2Col_ = np.zeros(LB_DEPTH, dtype=np.int32)
            for i in range(LB_DEPTH - 1):
                lb1[i, c] = lb1[i + 1, c]
                img1Col_[i] = lb1[i, c]
                lb2[i, c] = lb2[i + 1, c]
                img2Col_[i] = lb2[i, c]
            lb1[LB_DEPTH - 1, c] = frame0[r, c]
            img1Col_[LB_DEPTH - 1] = frame0[r, c]
            lb2[LB_DEPTH - 1, c] = frame1[r, c]
            img2Col_[LB_DEPTH - 1] = frame1[r, c]

            # --- computeSums ---
            csIxixL = np.int64(0)
            csIxiyL = np.int64(0)
            csIyiyL = np.int64(0)
            csDixL = np.int64(0)
            csDiyL = np.int64(0)
            if zIdx >= 0:
                csIxixL = csIxix[zIdx]
                csIxiyL = csIxiy[zIdx]
                csIyiyL = csIyiy[zIdx]
                csDixL = csDix[zIdx]
                csDiyL = csDiy[zIdx]

            # Top gradients (wrt = 1)
            # C integer division truncates toward zero; Python // floors.
            # Use int(x / 2) to match C behavior for negative values.
            wrt = 1
            cIxTopR = int(int(img1Col_[wrt]) - int(img1Win[wrt * 2 + 2 - 2]))
            cIxTopR = int(cIxTopR / 2) if cIxTopR >= 0 else -int((-cIxTopR) / 2)
            cIyTopR = int(int(img1Win[(wrt + 1) * 2 + 2 - 1]) - int(img1Win[(wrt - 1) * 2 + 2 - 1]))
            cIyTopR = int(cIyTopR / 2) if cIyTopR >= 0 else -int((-cIyTopR) / 2)
            delTopR = int(img1Win[wrt * 2 + 2 - 1]) - int(img2Win[wrt * 1 + 1 - 1])

            # Bottom gradients (wrb = WINDOW_SIZE - 1)
            wrb = WS - 1
            cIxBotR = int(int(img1Col_[wrb]) - int(img1Win[wrb * 2 + 2 - 2]))
            cIxBotR = int(cIxBotR / 2) if cIxBotR >= 0 else -int((-cIxBotR) / 2)
            cIyBotR = int(int(img1Win[(wrb + 1) * 2 + 2 - 1]) - int(img1Win[(wrb - 1) * 2 + 2 - 1]))
            cIyBotR = int(cIyBotR / 2) if cIyBotR >= 0 else -int((-cIyBotR) / 2)
            delBotR = int(img1Win[wrb * 2 + 2 - 1]) - int(img2Win[wrb * 1 + 1 - 1])

            # Column sum update
            csIxixR = cbIxix[nIdx] + cIxBotR * cIxBotR - cIxTopR * cIxTopR
            csIxiyR = cbIxiy[nIdx] + cIxBotR * cIyBotR - cIxTopR * cIyTopR
            csIyiyR = cbIyiy[nIdx] + cIyBotR * cIyBotR - cIyTopR * cIyTopR
            csDixR = cbDix[nIdx] + delBotR * cIxBotR - delTopR * cIxTopR
            csDiyR = cbDiy[nIdx] + delBotR * cIyBotR - delTopR * cIyTopR

            # Row sum (integral) update
            ixix_acc += (csIxixR - csIxixL)
            ixiy_acc += (csIxiyR - csIxiyL)
            iyiy_acc += (csIyiyR - csIyiyL)
            dix_acc += (csDixR - csDixL)
            diy_acc += (csDiyR - csDiyL)

            # --- computeFlow ---
            # HW uses float (32-bit) for all of det, i00..i11, fx, fy.
            f_ixix = np.float32(int(ixix_acc))
            f_iyiy = np.float32(int(iyiy_acc))
            f_ixiy = np.float32(int(ixiy_acc))
            f_dix = np.float32(int(dix_acc))
            f_diy = np.float32(int(diy_acc))

            det = np.float32(f_ixix * f_iyiy - f_ixiy * f_ixiy)
            if det <= np.float32(1.0) or r < WS or c < (WS + 1):
                fx_ = np.float32(0.0)
                fy_ = np.float32(0.0)
            else:
                i00 = np.float32(f_iyiy / det)
                i01 = np.float32(np.float32(-f_ixiy) / det)
                i10 = np.float32(np.float32(-f_ixiy) / det)
                i11 = np.float32(f_ixix / det)
                fx_ = np.float32(np.float32(i00 * f_dix) + np.float32(i01 * f_diy))
                fy_ = np.float32(np.float32(i10 * f_dix) + np.float32(i11 * f_diy))

            fx_out[r, c] = fx_
            fy_out[r, c] = fy_

            # Update img1Win and img2Win (shift left and insert new column)
            for i in range(LB_DEPTH):
                img1Win[i * 2] = img1Win[i * 2 + 1]
            for i in range(LB_DEPTH):
                img1Win[i * 2 + 1] = img1Col_[i]
                img2Win[i] = img2Col_[i]

            # Update column sum buffers
            cbIxix[nIdx] = csIxixR
            cbIxiy[nIdx] = csIxiyR
            cbIyiy[nIdx] = csIyiyR
            cbDix[nIdx] = csDixR
            cbDiy[nIdx] = csDiyR

            csIxix[nIdx] = csIxixR
            csIxiy[nIdx] = csIxiyR
            csIyiy[nIdx] = csIyiyR
            csDix[nIdx] = csDixR
            csDiy[nIdx] = csDiyR

            zIdx += 1
            if zIdx == cols:
                zIdx = 0
            nIdx += 1
            if nIdx == cols:
                nIdx = 0

        # Reset window state between rows (HW cleans up at end of
        # computeSums but the line buffer persists; the per-row reset
        # of accumulators and window state happens implicitly by the
        # HW re-entering the outer loop with static vars still set).
        # Actually in HW, img1Win and img2Win are NOT reset between rows.
        # They persist. The cleanup only happens after ALL rows are done.
        # So we should NOT reset them here.

    # Convert float32 to uint32 (reinterpret cast, matching writeOutput8)
    outx = fx_out.flatten().view(np.uint32)
    outy = fy_out.flatten().view(np.uint32)

    return outx, outy


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
