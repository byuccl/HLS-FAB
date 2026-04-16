import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from config/xf_config_params.h) ---
HEIGHT = 64
WIDTH = 64

# IN_TYPE = XF_8UC1 -> ap_uint<8>, uint8
# OUT_TYPE = XF_32UC1 -> ap_uint<32>, uint32
# INPUT_PTR_WIDTH = 32, OUTPUT_PTR_WIDTH = 32
# NPPCX = XF_NPPC1
# WINSIZE_OFLOW = 11
# NUM_LEVELS = 5
# NUM_ITERATIONS = 5
# TYPE_FLOW_WIDTH = 16, TYPE_FLOW_INT = 10 -> ap_fixed<16,10> (6 fractional bits)
# NUM_LINES_FINDIT = 50

WINSIZE = 11

# The function operates on one pyramid level per call.
# Inputs: two grayscale images (_current_img, _next_image), a packed flow
# input (_streamFlowin), and scalar control params.
# Output: packed flow output (_streamFlowout).
# Each flow pixel is uint32: upper 16 bits = U (horizontal flow as
# ap_fixed<16,10>), lower 16 bits = V (vertical flow as ap_fixed<16,10>).
#
# For csim testing we use init_flag=1, scale_up_flag=0, level=NUM_LEVELS-1=4.
# With init_flag=1, the input flow is read but discarded; new flow is
# computed from scratch.

function_signature = (
    "void pyr_dense_optical_flow_accel("
    "int32_t* _current_img, "
    "int32_t* _next_image, "
    "int32_t* _streamFlowin, "
    "int32_t* _streamFlowout)"
)

description = f"""
This HLS kernel computes one iteration of pyramidal Lucas-Kanade dense
optical flow at a single pyramid level.

Parameters:
    _current_img:   Pointer to a flat array of {HEIGHT * WIDTH} uint8
                    pixel values (packed 4 per 32-bit word) for the
                    current grayscale image ({HEIGHT}x{WIDTH}, row-major).
    _next_image:    Pointer to a flat array of {HEIGHT * WIDTH} uint8
                    pixel values (packed 4 per 32-bit word) for the next
                    grayscale image ({HEIGHT}x{WIDTH}, row-major).
    _streamFlowin:  Pointer to a flat array of {HEIGHT * WIDTH} uint32
                    packed flow vectors from a prior iteration. Each
                    element encodes horizontal (U) flow in the upper 16
                    bits and vertical (V) flow in the lower 16 bits, both
                    as signed fixed-point <16,10> values.
    _streamFlowout: Pointer to a flat array of {HEIGHT * WIDTH} uint32
                    packed flow vectors (same encoding as _streamFlowin).

Algorithm (single-level, single-iteration Lucas-Kanade):
  1. Compute spatial gradients Ix (forward difference in x) and Iy
     (forward difference in y) from the current image.
  2. Compute temporal gradient It = current_pixel - next_pixel (with
     init_flag=1, no warping is applied).
  3. Over an {WINSIZE}x{WINSIZE} sliding window, accumulate the
     structure tensor components: sum(Ix^2), sum(Iy^2), sum(IxIy),
     sum(IxIt), sum(IyIt).
  4. Solve the 2x2 linear system to find flow (U, V) at each pixel.
     If the determinant is near zero or the minimum eigenvalue is too
     small, set flow to zero.
  5. Apply a 3x3 median filter on the U and V flow fields separately,
     using replicate border padding.
  6. Pack U, V into 32-bit words (U in upper 16 bits, V in lower 16
     bits) and write to the output buffer.
"""

inputs_info = [
    {
        "name": "_current_img",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "_next_image",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "_streamFlowin",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 4294967295,
    },
    {
        "name": "_streamFlowout",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]


def _to_fixed16_10(val):
    """Convert float to ap_fixed<16,10> representation (as int16)."""
    v = val * 64.0  # 6 fractional bits
    v = np.clip(v, -32768, 32767)
    return np.int16(np.round(v))


def _pack_flow(u_fixed, v_fixed):
    """Pack two int16 flow values into a uint32 word.
    Upper 16 bits = U, lower 16 bits = V."""
    u16 = np.uint16(u_fixed.view(np.uint16))
    v16 = np.uint16(v_fixed.view(np.uint16))
    return (np.uint32(u16) << 16) | np.uint32(v16)


def behavioral(_current_img, _next_image, _streamFlowin):
    """Compute one iteration of LK dense optical flow (init_flag=1).

    Closely matches the Vitis HLS xf::cv::densePyrOpticalFlow kernel
    with init_flag=1, scale_up_flag=0, using integer/fixed-point arithmetic
    where the HLS does so, and float32/half where the HLS uses those types.
    """
    rows, cols = HEIGHT, WIDTH
    curr = _current_img.reshape((rows, cols)).astype(np.int32)
    nxt = _next_image.reshape((rows, cols)).astype(np.int32)

    # ---- Step 1: Spatial and temporal gradients (ap_int<9>) ----
    # Ix: j=0 no output, j=1: pix[1]-pix[0], j>=2: pix[j]-pix[j-2],
    #      j=cols (extra): pix[cols-1]-pix[cols-2].
    # Iy: i=0: row[1]-row[0], i=rows-1: row[rows-1]-row[rows-2],
    #      else: row[i+1]-row[i-1].
    # It: curr_px - nextI (with init_flag=1, u=v=0, nextI=next[i][j]).
    # All clipped to ap_int<9> range [-256, 255].

    Ix = np.zeros((rows, cols), dtype=np.int32)
    Iy = np.zeros((rows, cols), dtype=np.int32)
    It = np.zeros((rows, cols), dtype=np.int32)

    for i in range(rows):
        if i == 0:
            Iy[i, :] = curr[1, :] - curr[0, :]
        elif i == rows - 1:
            Iy[i, :] = curr[rows - 1, :] - curr[rows - 2, :]
        else:
            Iy[i, :] = curr[i + 1, :] - curr[i - 1, :]

        Ix[i, 0] = curr[i, 1] - curr[i, 0]
        for j in range(2, cols):
            Ix[i, j - 1] = curr[i, j] - curr[i, j - 2]
        Ix[i, cols - 1] = curr[i, cols - 1] - curr[i, cols - 2]

    # It: temporal gradient (ap_fixed<17,9> with 8 fractional bits,
    # but with init_flag=1 and u=v=0 the lookup is exact integer)
    It = curr - nxt

    # ---- Step 2: Structure tensor windowed sums ----
    half_win = WINSIZE // 2

    # Pad with zeros at bottom and right for out-of-range reads
    Ix_pad = np.pad(Ix, ((0, half_win), (0, half_win)),
                    mode='constant', constant_values=0).astype(np.int64)
    Iy_pad = np.pad(Iy, ((0, half_win), (0, half_win)),
                    mode='constant', constant_values=0).astype(np.int64)
    It_pad = np.pad(It, ((0, half_win), (0, half_win)),
                    mode='constant', constant_values=0).astype(np.int64)

    ext_h = rows + half_win
    ext_w = cols + half_win

    # Column sums of products (exact integer arithmetic matching ap_fixed)
    def col_sum_2d(A, B, winsize, ext_h, ext_w):
        cs = np.zeros((ext_h, ext_w), dtype=np.int64)
        for i in range(ext_h):
            prod = A[i, :ext_w] * B[i, :ext_w]
            if i == 0:
                cs[i] = prod
            elif i < winsize:
                cs[i] = cs[i - 1] + prod
            else:
                old_prod = A[i - winsize, :ext_w] * B[i - winsize, :ext_w]
                cs[i] = cs[i - 1] + prod - old_prod
        return cs

    cs_ixix = col_sum_2d(Ix_pad, Ix_pad, WINSIZE, ext_h, ext_w)
    cs_iyiy = col_sum_2d(Iy_pad, Iy_pad, WINSIZE, ext_h, ext_w)
    cs_ixiy = col_sum_2d(Ix_pad, Iy_pad, WINSIZE, ext_h, ext_w)
    cs_ixit = col_sum_2d(Ix_pad, It_pad, WINSIZE, ext_h, ext_w)
    cs_iyit = col_sum_2d(Iy_pad, It_pad, WINSIZE, ext_h, ext_w)

    # Row sums of column sums
    def row_sum_2d(cs, winsize, ext_h, ext_w):
        rs = np.zeros((ext_h, ext_w), dtype=np.int64)
        for j in range(ext_w):
            if j == 0:
                rs[:, j] = cs[:, j]
            elif j < winsize:
                rs[:, j] = rs[:, j - 1] + cs[:, j]
            else:
                rs[:, j] = rs[:, j - 1] + cs[:, j] - cs[:, j - winsize]
        return rs

    rs_ixix = row_sum_2d(cs_ixix, WINSIZE, ext_h, ext_w)
    rs_iyiy = row_sum_2d(cs_iyiy, WINSIZE, ext_h, ext_w)
    rs_ixiy = row_sum_2d(cs_ixiy, WINSIZE, ext_h, ext_w)
    rs_ixit = row_sum_2d(cs_ixit, WINSIZE, ext_h, ext_w)
    rs_iyit = row_sum_2d(cs_iyit, WINSIZE, ext_h, ext_w)

    # Extract output pixels and apply shifts (>>2 for spatial, >>1 for cross)
    # These are integer right shifts (floor division for positive, but
    # arithmetic shift for negative).
    raw_Ix2 = rs_ixix[half_win:half_win + rows, half_win:half_win + cols]
    raw_Iy2 = rs_iyiy[half_win:half_win + rows, half_win:half_win + cols]
    raw_IxIy = rs_ixiy[half_win:half_win + rows, half_win:half_win + cols]
    raw_IxIt = rs_ixit[half_win:half_win + rows, half_win:half_win + cols]
    raw_IyIt = rs_iyit[half_win:half_win + rows, half_win:half_win + cols]

    # Arithmetic right shift (>> in C for signed)
    sIx2 = raw_Ix2 >> 2
    sIy2 = raw_Iy2 >> 2
    sIxIy = raw_IxIy >> 2
    sIxIt = raw_IxIt >> 1
    sIyIt = raw_IyIt >> 1

    # ---- Step 3: Solve for flow (find_flow) ----
    # The HLS uses fixed-point for determinant and float32 for eigenvalue.
    flowU = np.zeros((rows, cols), dtype=np.float64)
    flowV = np.zeros((rows, cols), dtype=np.float64)

    # Determinant in fixed-point (exact integer since sIx2 etc. are integers)
    det = sIx2 * sIy2 - sIxIy * sIxIy

    # Eigenvalue computation (HLS uses float32 and half precision)
    # S12sq = IxIy^2, then S12sq = S12sq*4 + S1122^2
    S1122 = sIx2 + sIy2
    S12sq_fixpt = sIxIy * sIxIy
    S12sq_fixpt = S12sq_fixpt * 4 + S1122 * S1122

    # Cast to float32 for eigenvalue test (matching HLS)
    S1122_f = S1122.astype(np.float32)
    S12sq_f = S12sq_fixpt.astype(np.float32)
    eig_comp = S1122_f - np.sqrt(np.maximum(S12sq_f, np.float32(0)))
    div_by_eig = np.float32(np.float16(1.0 / (2.0 * WINSIZE * WINSIZE)))
    eig_comp2 = eig_comp * div_by_eig
    eig_comp3 = np.abs(eig_comp2)

    valid = (det != 0) & (eig_comp3 >= np.float32(0.025))

    # Flow computation: the HLS uses ap_fixed<40,2> for 1/det and wide
    # fixed-point for the numerator. We quantize divideBy to match
    # ap_fixed<40,2> (38 fractional bits).
    safe_det = np.where(valid, det.astype(np.float64), 1.0)
    # Quantize 1/det to ap_fixed<40,2> precision (38 fractional bits)
    divideBy = 1.0 / safe_det
    divideBy = np.floor(divideBy * (2.0**38)) / (2.0**38)

    sIx2_f = sIx2.astype(np.float64)
    sIy2_f = sIy2.astype(np.float64)
    sIxIy_f = sIxIy.astype(np.float64)
    sIxIt_f = sIxIt.astype(np.float64)
    sIyIt_f = sIyIt.astype(np.float64)

    numU = sIy2_f * sIxIt_f - sIxIy_f * sIyIt_f
    numV = sIx2_f * sIyIt_f - sIxIy_f * sIxIt_f
    flowU = np.where(valid, numU * divideBy, 0.0)
    flowV = np.where(valid, numV * divideBy, 0.0)

    # ---- Step 4: Quantize flow to ap_fixed<16,10> BEFORE median ----
    # The HLS stores flow as ap_fixed<16,10> (6 fractional bits) and the
    # median filter operates on this quantized representation.
    flowU_q = np.floor(flowU * 64.0) / 64.0
    flowV_q = np.floor(flowV * 64.0) / 64.0

    # ---- Step 5: Median blur (WINDOW_SIZE_FL=5) ----
    MED_WIN = 5

    def median_blur(field, win):
        h = win // 2
        padded = np.pad(field, h, mode='edge')
        out = np.zeros_like(field)
        for i in range(field.shape[0]):
            for j in range(field.shape[1]):
                window = padded[i:i + win, j:j + win].flatten()
                out[i, j] = np.median(window)
        return out

    flowU_filt = median_blur(flowU_q, MED_WIN)
    flowV_filt = median_blur(flowV_q, MED_WIN)

    # ---- Step 6: Quantize final result and pack ----
    u_fixed = _to_fixed16_10(flowU_filt)
    v_fixed = _to_fixed16_10(flowV_filt)
    result = _pack_flow(u_fixed, v_fixed)

    return result.flatten()


examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
