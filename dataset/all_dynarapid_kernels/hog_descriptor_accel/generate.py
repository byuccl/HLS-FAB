import numpy as np
import json
import os
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
WIN_HEIGHT = 128
WIN_WIDTH = 64
WIN_STRIDE = 8
BLOCK_HEIGHT = 16
BLOCK_WIDTH = 16
CELL_HEIGHT = 8
CELL_WIDTH = 8
NO_OF_BINS = 9
NOVCPB = BLOCK_HEIGHT // CELL_HEIGHT  # 2
NOHCPB = BLOCK_WIDTH // CELL_WIDTH    # 2
NOBPB = NO_OF_BINS * NOHCPB * NOVCPB  # 36
NOVB = (HEIGHT // CELL_HEIGHT) - 1    # 15
NOHB = (WIDTH // CELL_WIDTH) - 1      # 15
NOHC = WIDTH // CELL_WIDTH            # 16
DESC_SIZE = (NOVB * NOHB * NOBPB) >> 1  # 4050

function_signature = "void hog_descriptor_accel(int32_t* img_in, int32_t* desc_out)"

description = f"""Computes HOG (Histogram of Oriented Gradients) descriptors for a {HEIGHT}x{WIDTH} grayscale image.

The input is a flat array of {HEIGHT*WIDTH} uint8 pixel values representing an 8-bit grayscale image.

The function outputs {DESC_SIZE} uint32 values, each packing two 16-bit fixed-point (Q0.16) normalized
histogram bin values.

The HOG pipeline consists of:
1. Gradient computation using [-1, 0, 1] filters for X and Y directions with zero-padded borders.
   The output gradient size is {HEIGHT}x{WIDTH} (same as input), with border pixels having zero gradients.
2. Phase (angle) computation using a LUT-based atan2, producing values in Q9.7 fixed-point format
   representing angles in degrees [0, 360).
3. Magnitude computation as sqrt(gx^2 + gy^2) in Q9.7 format.
4. Histogram binning: pixels are grouped into {CELL_HEIGHT}x{CELL_WIDTH} cells. For each cell, a 9-bin
   histogram (bin stride = 20 degrees) is computed using bilinear interpolation of magnitude across
   adjacent bins.
5. Block normalization: overlapping {BLOCK_HEIGHT}x{BLOCK_WIDTH} blocks (2x2 cells) are L2-normalized
   with clipping at 0.2 and re-normalization.
6. Non-Repetitive Block (NRB) output: {NOVB}x{NOHB} = {NOVB*NOHB} blocks, each with {NOBPB} bins
   ({NO_OF_BINS} bins x {NOHCPB} x {NOVCPB} cells). Each bin is a 16-bit value. Two bins are packed
   per 32-bit output word, producing {DESC_SIZE} total uint32 values."""

inputs_info = [
    {
        "name": "img_in",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "desc_out",
        "array_dimension": 1,
        "shape": (DESC_SIZE,),
        "total_elements": DESC_SIZE,
        "dtype": "int32",
        "iotype": "out",
    },
]

# Load LUT tables
_lut_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "luts.json")
with open(_lut_path, "r") as _f:
    _luts = json.load(_f)
_division_lut8 = np.array(_luts["division_lut8"], dtype=np.uint16)
_base_addr_lut8 = np.array(_luts["base_addr_lut8"], dtype=np.int32)
_atan_lut_fp8 = np.array(_luts["atan_lut_fp8"], dtype=np.int16)
_division_lut = np.array(_luts["division_lut"], dtype=np.uint16)

XF_PI_FIXED = 12867
XF_PI_BY_2_FIXED = 6433
XF_NORM_FACTOR = 58671
XF_HOG_1_BY_20 = 3276
XF_HOG_PI = 23040
XF_CLIP_VAL = 13108
XF_MAX_VAL_16_BIT = 65535
XF_3_P_6 = 921
DATA_SIZE = 16
LUT_BITS = 11


def _inverse8(x):
    """Inverse8: returns (inv_val, N). x is uint8."""
    if x == 0:
        return int(_division_lut8[0]), 15
    return int(_division_lut8[x]), 15


def _divide_y_by_x8(x, y):
    """DivideYByX8: x, y are uint8. Returns (div_val, MOut, NOut)."""
    x_inv, n_out = _inverse8(x)
    m_out = (16 - n_out) + 8
    n_out2 = n_out + (8 - 8)
    div_val = int(y) * x_inv
    return div_val, m_out, n_out2


def _atan2_lut8(xs, ys):
    """Atan2LUT8: xs, ys are signed 9-bit (int16). Returns theta in Q4.12."""
    xs = np.int16(xs)
    ys = np.int16(ys)
    if ys == 0 and xs == 0:
        return 0

    gx_fp = int(abs(xs))
    gy_fp = int(abs(ys))
    s1 = 1 if xs >= 0 else -1
    s2 = 1 if ys >= 0 else -1
    s = s1 * s2

    # DivideYByX8 with uint8 inputs
    gx_u8 = gx_fp & 0xFF
    gy_u8 = gy_fp & 0xFF

    div_val, m_out, n_out = _divide_y_by_x8(gx_u8, gy_u8)

    frac_extractor = (1 << n_out) - 1
    int_part = (div_val >> n_out) & 0xFFFF

    if int_part < 63:
        frac_bits = _base_addr_lut8[63 + int_part]
        frac_part = div_val & frac_extractor
        shift = n_out - frac_bits
        if shift >= 0:
            frac_ind = frac_part >> shift
        else:
            frac_ind = frac_part << (-shift)
        ind = _base_addr_lut8[int_part] + frac_ind
        if ind < 0:
            ind = 0
        if ind >= len(_atan_lut_fp8):
            ind = len(_atan_lut_fp8) - 1
        theta = int(_atan_lut_fp8[ind])
    else:
        theta = XF_PI_BY_2_FIXED

    if gx_u8 == 0:
        theta = XF_PI_BY_2_FIXED

    if s < 0:
        theta = -theta

    if xs > 0:
        return theta
    elif ys >= 0 and xs < 0:
        return XF_PI_FIXED + theta
    elif ys < 0 and xs < 0:
        return theta - XF_PI_FIXED
    elif ys > 0 and xs == 0:
        return XF_PI_BY_2_FIXED
    elif ys < 0 and xs == 0:
        return -XF_PI_BY_2_FIXED
    return theta


def _sqrt_hog(x_val, obn):
    """Fixed-point sqrt matching xFSqrtHOG. x_val is unsigned integer, treated
    as fixed-point. obn is output bit width."""
    # We use integer arithmetic to match the HLS implementation
    if x_val == 0:
        return 0
    # Use floating point sqrt and convert back
    # The HLS function uses iterative digit-by-digit sqrt
    # For correctness we replicate the algorithm
    result = int(np.floor(np.sqrt(float(x_val))))
    return result & ((1 << obn) - 1)


def _inverse16(x, m):
    """Inverse function for 16-bit input. Returns (val, N)."""
    if x == 0:
        return int(_division_lut[0]), m + DATA_SIZE - 1

    x = int(x) & 0xFFFF
    x1 = (x >> 12) & 0xF
    x2 = (x >> 8) & 0xF
    x3 = (x >> 4) & 0xF
    x4 = x & 0xF

    if x1 > 0:
        block = 0
        x_sel = x1
    elif x2 > 0:
        block = 1
        x_sel = x2
    elif x3 > 0:
        block = 2
        x_sel = x3
    else:
        block = 3
        x_sel = x4

    if (x_sel >> 3) > 0:
        pos = 1
    elif (x_sel >> 2) > 0:
        pos = 2
    elif (x_sel >> 1) > 0:
        pos = 3
    elif x_sel > 0:
        pos = 4
    else:
        pos = 0

    b_l = 4 * block + pos
    shift = DATA_SIZE - LUT_BITS  # 16 - 11 = 5
    index = ((x << (b_l - 1)) & 0x7FFF) >> shift
    if index >= len(_division_lut):
        index = len(_division_lut) - 1
    val = int(_division_lut[index])
    n = m + DATA_SIZE - b_l
    return val, n


def _identify_sign_bits_24(in_val):
    """Count leading sign bits in 24-bit value."""
    in_val = in_val & 0xFFFFFF
    signbit = (in_val >> 23) & 1
    counter = 0
    for i in range(24):
        bit_val = (in_val >> (23 - counter)) & 1
        if bit_val == signbit:
            counter += 1
        else:
            break
    return counter


def _inverse24(in_val):
    """xFInverse24: 24-bit inverse. Returns (out_val, n)."""
    in_val = int(in_val) & 0xFFFFFF
    m = 16
    s = _identify_sign_bits_24(in_val)
    catch_val = (in_val << s) & 0xFFFFFF
    m1 = m - s
    in_val_shifted = (catch_val >> 8) & 0xFFFF
    out_val, n2 = _inverse16(in_val_shifted, m1)
    n = n2
    tmp_n = n2 - s
    if tmp_n < 0:
        tmp_n_2 = -tmp_n
        out_val = out_val << tmp_n_2
        n = n2 + tmp_n_2
    return out_val, n


def behavioral(img_in):
    img = img_in.astype(np.uint8).reshape(HEIGHT, WIDTH)

    # Step 1: Gradient computation with zero-padded borders
    # Pad image with zeros: top, bottom rows; left, right columns
    padded = np.zeros((HEIGHT + 2, WIDTH + 2), dtype=np.int16)
    padded[1:-1, 1:-1] = img.astype(np.int16)

    # Gradient: gx[r,c] = padded[r+1, c+2] - padded[r+1, c]
    #           gy[r,c] = padded[r+2, c+1] - padded[r, c+1]
    # Output for rows 1..HEIGHT-1 (row 0 is not computed since first row reads are special)
    # Actually the gradient kernel processes rows 1 to HEIGHT (producing HEIGHT gradients)
    # Looking at the code: row loop goes from 1 to height-1, then handles last row with zero-padded bottom
    # The gradient output has HEIGHT rows and WIDTH columns

    gx = np.zeros((HEIGHT, WIDTH), dtype=np.int16)
    gy = np.zeros((HEIGHT, WIDTH), dtype=np.int16)

    # Row 0: all zero (first row buffer is zeroed, gradient will be zero)
    # Actually, looking at the code more carefully:
    # - buf[0] is initialized to 0
    # - buf[1] = first row of input
    # - Row_Loop runs from row=1 to height-1, reading new row each time
    # - Then processes last row with zero-padded bottom
    # For row=1: tp=0(zeros), mid=1(row0), bottom=2(row1)
    #   gx = row0[j+2] - row0[j] (uses mid buffer, columns j and j+2)
    #   gy = row1[j+1] - 0[j+1] = row1[j+1]
    # Wait, the gradient output starts from i=1 (Row_Loop starts at 1)
    # Each row of gradient corresponds to row (i-1) of the output? No...
    # Actually gradient row i-1 is output during Row_Loop iteration for row i
    # Let me re-read the code...

    # The gradient kernel produces HEIGHT output rows:
    # For row=1 to HEIGHT-1: compute gradient of the 'mid' row
    # Then for the last iteration with zero bottom: compute gradient of last row
    # So output gradient[0] = gradient of row 0 (with zero top)
    # gradient[i] = gradient of row i (with row i-1 above, row i+1 below)
    # gradient[HEIGHT-1] = gradient of last row (with zero bottom)

    for i in range(HEIGHT):
        for j in range(WIDTH):
            # X gradient: pixel[i][j+1] - pixel[i][j-1]
            right = int(img[i, j+1]) if j+1 < WIDTH else 0
            left = int(img[i, j-1]) if j-1 >= 0 else 0
            gx[i, j] = right - left

            # Y gradient: pixel[i+1][j] - pixel[i-1][j]
            bottom = int(img[i+1, j]) if i+1 < HEIGHT else 0
            top = int(img[i-1, j]) if i-1 >= 0 else 0
            gy[i, j] = bottom - top

    # Step 2: Phase and Magnitude computation (per pixel)
    phase = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)
    mag = np.zeros((HEIGHT, WIDTH), dtype=np.uint16)

    for i in range(HEIGHT):
        for j in range(WIDTH):
            gx_val = int(gx[i, j])
            gy_val = int(gy[i, j])

            # Phase: atan2 in Q4.12 -> convert to degrees in Q9.7
            ret = _atan2_lut8(gx_val, gy_val)
            if ret < 0 or (ret == 0 and gy_val < 0):
                result_temp = ret + XF_PI_FIXED + XF_PI_FIXED
            else:
                result_temp = ret
            result_phase = (XF_NORM_FACTOR * result_temp) >> 15  # Q9.7
            phase[i, j] = result_phase & 0xFFFF

            # Magnitude
            gx_abs = abs(gx_val) & 0xFF  # cast to uint8
            gy_abs = abs(gy_val) & 0xFF
            gx_sq = gx_abs * gx_abs
            gy_sq = gy_abs * gy_abs
            sum_sq = gx_sq + gy_sq
            tmp_mag = sum_sq << 14
            result_mag = _sqrt_hog(tmp_mag, 16)
            mag[i, j] = result_mag & 0xFFFF

    # Step 3: Histogram computation per cell row
    # Cell dimensions: CELL_HEIGHT x CELL_WIDTH = 8x8
    # Number of horizontal cells: NOHC = 16
    # Number of vertical cell rows: HEIGHT / CELL_HEIGHT = 16

    bin_stride = 180 // NO_OF_BINS  # 20
    bin_stride_fix = bin_stride << 7  # 2560 in Q9.7

    # Bin centers in Q9.7
    bin_center = np.zeros(NO_OF_BINS, dtype=np.uint16)
    offset = (bin_stride_fix) >> 1  # 1280
    for b in range(NO_OF_BINS):
        bin_center[b] = offset
        offset += bin_stride_fix

    # Compute histogram arrays for each cell row
    # HA[cell_row][cell_col][bin] - histogram in Q15.8 format
    num_cell_rows = HEIGHT // CELL_HEIGHT  # 16
    HA = np.zeros((num_cell_rows, NOHC, NO_OF_BINS), dtype=np.int64)
    ssv = np.zeros((num_cell_rows, NOHC), dtype=np.int64)

    for cr in range(num_cell_rows):
        # Initialize HA for this cell row
        for cc in range(NOHC):
            for b in range(NO_OF_BINS):
                HA[cr, cc, b] = 0

        for ci in range(CELL_HEIGHT):
            row = cr * CELL_HEIGHT + ci
            for cc in range(NOHC):
                bins_tmp = np.zeros(NO_OF_BINS, dtype=np.int64)
                for cj in range(CELL_WIDTH):
                    col = cc * CELL_WIDTH + cj
                    p = int(phase[row, col])
                    m_val = int(mag[row, col])

                    # Scale to PI (if > 180 in Q9.7)
                    if p > XF_HOG_PI:
                        p = p - XF_HOG_PI

                    # Bilinear interpolation
                    p = int(p)
                    m_val = int(m_val)
                    tmp_bin_idx = p * XF_HOG_1_BY_20  # Q9.7 x Q0.16 -> Q9.23
                    bin_idx = (tmp_bin_idx >> 23) & 0xFF
                    if bin_idx >= NO_OF_BINS:
                        bin_idx = NO_OF_BINS - 1

                    mag_factor = m_val * XF_HOG_1_BY_20  # Q9.23

                    bc = int(bin_center[bin_idx])
                    if p < bc:
                        if bin_idx == 0:
                            frac = p - (bc - bin_stride_fix)
                            index_1 = bin_idx
                            index_2 = NO_OF_BINS - 1
                        else:
                            frac = p - int(bin_center[bin_idx - 1])
                            index_1 = bin_idx
                            index_2 = bin_idx - 1
                    else:
                        if bin_idx == NO_OF_BINS - 1:
                            frac = (bc + bin_stride_fix) - p
                            index_1 = bin_idx
                            index_2 = 0
                        else:
                            frac = int(bin_center[bin_idx + 1]) - p
                            index_1 = bin_idx
                            index_2 = bin_idx + 1

                    frac = int(frac) & 0xFFFF  # uint16
                    # part_1: Q9.7 x Q9.23 -> Q18.30 >> 22 -> Q18.8
                    part_1 = (int(frac) * int(mag_factor)) >> 22
                    m_shifted = int(m_val) << 1  # Q9.8
                    part_2 = m_shifted - part_1

                    bins_tmp[index_1] += part_1
                    bins_tmp[index_2] += part_2

                for b in range(NO_OF_BINS):
                    HA[cr, cc, b] += bins_tmp[b]

                # Compute ssv
                tmp_ssv = 0
                for b in range(NO_OF_BINS):
                    tmp_ssv += HA[cr, cc, b] * HA[cr, cc, b]
                ssv[cr, cc] = tmp_ssv

    # Step 4: Block normalization (NRB mode)
    # Process blocks: each block spans 2 consecutive cell rows and 2 consecutive cell columns
    # NOVB = 15, NOHB = 15
    # For each vertical block index bi, we use cell rows (bi, bi+1) = (HA_1, HA_2)

    desc_output = []  # list of uint32 values

    for bi in range(NOVB):
        cr1 = bi
        cr2 = bi + 1

        for bj in range(NOHB):
            # Normalization step 1: compute norm factor from ssv
            tmp_nf_1 = ssv[cr1, bj] + ssv[cr2, bj]
            tmp_nf_2 = ssv[cr1, bj + 1] + ssv[cr2, bj + 1]
            tmp_nf_sum = tmp_nf_1 + tmp_nf_2
            if tmp_nf_sum < 0:
                tmp_nf_sum = 0

            # sqrt of tmp_nf_sum (Q16.8 result) + epsilon
            tmp_nf_sq24 = int(np.floor(np.sqrt(float(tmp_nf_sum)))) + XF_3_P_6

            # Inverse of tmp_nf_sq24 (24-bit)
            if tmp_nf_sq24 > 0:
                nf, n = _inverse24(tmp_nf_sq24)
            else:
                nf, n = 0xFFFFFFFF, 32

            # Normalization step 2: normalize and clip
            norm_block = np.zeros(NOBPB, dtype=np.uint32)
            fx_clip_val = XF_CLIP_VAL  # 0.2 in Q0.16

            # Block contains 2x2 cells:
            # offset layout: cell(cr1,bj) -> offset 0..8
            #                cell(cr1,bj+1) -> offset 9..17
            #                cell(cr2,bj) -> offset 18..26
            #                cell(cr2,bj+1) -> offset 27..35
            # Block cell layout matches C++ normalization:
            # offset_1[0]=0: HA_1[bj] (cr1, bj)
            # offset_2[0]=9: HA_2[bj] (cr2, bj)
            # offset_1[1]=18: HA_1[bj+1] (cr1, bj+1)
            # offset_2[1]=27: HA_2[bj+1] (cr2, bj+1)
            offset_map = [
                (cr1, bj, 0),
                (cr2, bj, NO_OF_BINS),
                (cr1, bj + 1, NO_OF_BINS * 2),
                (cr2, bj + 1, NO_OF_BINS * 3),
            ]

            rnf_sum = 0
            for cr, cc, off in offset_map:
                for b in range(NO_OF_BINS):
                    ha_val = int(HA[cr, cc, b])
                    shift_amt = n - 8
                    if shift_amt >= 0:
                        tmp_clip = (ha_val * nf) >> shift_amt
                    else:
                        tmp_clip = (ha_val * nf) << (-shift_amt)
                    tmp_clip = tmp_clip & 0xFFFFFFFF  # uint32
                    if tmp_clip > fx_clip_val:
                        tmp_clip = fx_clip_val
                    norm_block[off + b] = tmp_clip
                    rnf_sum += tmp_clip * tmp_clip

            # Re-normalization
            rnf_sum = rnf_sum & 0xFFFFFFFF  # limit to 32 bits
            fx_rnf_sq = int(np.floor(np.sqrt(float(rnf_sum))))
            fx_rnf_sq = (fx_rnf_sq >> 1) & 0xFFFF  # Q1.15

            # Inverse of fx_rnf_sq
            if fx_rnf_sq > 0:
                rnf, n_rnf = _inverse16(fx_rnf_sq, 1)
            else:
                rnf, n_rnf = 0xFFFF, 16

            # Apply re-normalization and pack output
            block_data = np.zeros(NOBPB, dtype=np.uint16)
            for k in range(NOBPB):
                if n_rnf >= 0:
                    tmp = (int(norm_block[k]) * rnf) >> n_rnf
                else:
                    tmp = (int(norm_block[k]) * rnf) << (-n_rnf)
                if tmp > XF_MAX_VAL_16_BIT:
                    tmp = XF_MAX_VAL_16_BIT
                block_data[k] = tmp & 0xFFFF

            # NRB output: pack 2 x 16-bit values per 32-bit word
            # loop_count = NOBPB * 2 / 4 = 36 * 2 / 4 = 18
            # Each 576-bit block_data is packed as: 36 x 16-bit values
            # Written as 18 x 32-bit words (2 bins per word)
            for k in range(0, NOBPB, 2):
                lo = int(block_data[k])
                hi = int(block_data[k + 1])
                word = (hi << 16) | lo
                desc_output.append(word)

    result = np.array(desc_output, dtype=np.uint32)
    return result


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
