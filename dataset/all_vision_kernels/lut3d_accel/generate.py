import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 1080
WIDTH = 1920
LUT_DIM = 33

function_signature = (
    "void lut3d_accel("
    "ap_uint<8>* img_in, "
    "ap_uint<8>* img_out, "
    "ap_uint<32>* lut)"
)

description = f"""3D LUT color-mapping kernel that applies a {LUT_DIM}x{LUT_DIM}x{LUT_DIM} 3D lookup table
to a {WIDTH}x{HEIGHT} 3-channel 8-bit image using trilinear interpolation with Vitis HLS fixed-point
arithmetic.

Parameters:
    img_in:  Pointer to flat array of {HEIGHT * WIDTH * 3} uint8 values representing
             a {HEIGHT}x{WIDTH} image in row-major, channel-interleaved order (ch0,ch1,ch2,...).
    img_out: Pointer to flat array of {HEIGHT * WIDTH * 3} uint8 values for the output image,
             same format as input.
    lut:     Pointer to flat array of {LUT_DIM**3 * 3} uint32 values representing the 3D LUT.
             The LUT is stored as {LUT_DIM}x{LUT_DIM}x{LUT_DIM} entries in [z][y][x] order,
             with 3 float32 values per entry (ch0, ch1, ch2) bit-cast to uint32.
             Each float32 LUT value is in [0.0, 1.0] range.

Fixed-Point Types (from xf_3dlut.hpp):
    _FIXED_LUT_TYPE   = ap_ufixed<9,1>:  9 total bits, 1 integer bit, 8 fractional bits.
                        Range [0, ~2.0), precision 1/256. Used for LUT entry values and
                        distance fractions.
    _FIXED_PIXEL_TYPE = ap_ufixed<16,6>: 16 total bits, 6 integer bits, 10 fractional bits.
                        Range [0, ~64.0), precision 1/1024. Used for intermediate computation.
    _FIXED_OUT_PIXEL_TYPE = ap_ufixed<9,8>: 9 total bits, 8 integer bits, 1 fractional bit.
                        Used for final output scaling.

Algorithm (matching Vitis HLS fixed-point behaviour exactly):
    1. For each pixel, extract 3 channels (each 0-255):
       ch0 (R) from bits[7:0], ch1 (G) from bits[15:8], ch2 (B) from bits[23:16].
    2. Compute scale_i = ch_i / 255.0, quantized to ap_ufixed<9,1> (truncate to 8 frac bits):
       scale_i_fp = floor(ch_i / 255.0 * 256) / 256
    3. Compute index_i = scale_i_fp * (LUT_DIM - 1), quantized to ap_ufixed<16,6> (10 frac bits):
       index_i_fp = floor(scale_i_fp * 32 * 1024) / 1024
    4. Integer index: pixelIndex_i = floor(index_i_fp) (cast to int, dropping fractional bits)
    5. Distance fraction: dist_i = index_i_fp - pixelIndex_i, quantized to ap_ufixed<9,1>
       (8 frac bits, truncate): dist_i_fp = floor((index_i_fp - pixelIndex_i) * 256) / 256
    6. Border cases (priority: R-border, then G-border, then B-border):
       If pixelIndex_R + 1 == LUT_DIM: use direct LUT[B][G] (R-border)
       Elif pixelIndex_G + 1 == LUT_DIM: use direct LUT[B][R] (G-border)
       Elif pixelIndex_B + 1 == LUT_DIM: use direct LUT[G][R] (B-border)
    7. Otherwise: trilinear interpolation using interp3(), with all intermediate values in
       ap_ufixed<16,6> (10 fractional bits, truncate at each step).
       interp1(v1, v2, t) = v1 + t * (v2 - v1), result truncated to 10 frac bits.
       interp3(cube, dr, dg, db):
         a = interp1(P000, P100, dr)
         b = interp1(P001, P101, dr)
         c = interp1(P010, P110, dr)
         d = interp1(P011, P111, dr)
         e = interp1(a, b, dg)
         f = interp1(c, d, dg)
         g = interp1(e, f, db)
    8. Output = floor(result * 255), stored as ap_ufixed<9,8> (1 frac bit), then cast to uint8.
       Equivalent to: out = int(result_fp10 * 255 * 2) >> 1 (where result_fp10 has 10 frac bits).
"""

inputs_info = [
    {
        "name": "img_in",
        "array_dimension": 2,
        "shape": (HEIGHT * WIDTH, 3),
        "dtype": "uint8",
        "total_elements": HEIGHT * WIDTH * 3,
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT * WIDTH, 3),
        "dtype": "uint8",
        "total_elements": HEIGHT * WIDTH * 3,
        "iotype": "out",
    },
    {
        "name": "lut",
        "array_dimension": 1,
        "shape": (LUT_DIM ** 3 * 3,),
        "dtype": "uint32",
        "total_elements": LUT_DIM ** 3 * 3,
        "iotype": "in",
        "min": 0,
        "max": 0x3F800000,  # float 1.0 as uint32
    },
]


def fp9_1(x):
    """Quantize to ap_ufixed<9,1>: 8 fractional bits, truncate (floor toward 0)."""
    # Multiply by 2^8=256, truncate to non-negative integer, divide back
    return np.floor(np.clip(x, 0.0, None) * 256.0).astype(np.int64) / 256.0


def fp16_6(x):
    """Quantize to ap_ufixed<16,6>: 10 fractional bits, truncate (floor toward 0)."""
    # Multiply by 2^10=1024, truncate, divide back
    return np.floor(np.clip(x, 0.0, None) * 1024.0).astype(np.int64) / 1024.0


def interp1_fp(v1, v2, t):
    """
    Linear interpolation matching HLS interp1().
    All inputs are ap_ufixed<16,6> (10 frac bits); result is truncated to 10 frac bits.
    v1, v2, t are float64 arrays already quantized to 10 frac bits.
    """
    # HLS: ret = v1 + t * (v2 - v1)
    # The subtraction (v2-v1) can be negative; the product and addition use ap_ufixed<16,6>
    # In ap_ufixed arithmetic, intermediate negatives wrap or are handled by fixed-point rules.
    # We simulate with float64 then truncate to 10 frac bits (floor toward 0, clamp to 0).
    result = v1 + t * (v2 - v1)
    return fp16_6(result)


def interp3_fp(p000, p001, p010, p011, p100, p101, p110, p111, dr, dg, db):
    """
    Trilinear interpolation matching HLS interp3().
    Cube corners are in ap_ufixed<9,1> (8 frac bits), dr/dg/db in ap_ufixed<9,1>.
    Intermediate results in ap_ufixed<16,6> (10 frac bits).
    """
    a = interp1_fp(p000, p100, dr)
    b_ = interp1_fp(p001, p101, dr)
    c = interp1_fp(p010, p110, dr)
    d = interp1_fp(p011, p111, dr)

    e = interp1_fp(a, b_, dg)
    f = interp1_fp(c, d, dg)

    g = interp1_fp(e, f, db)
    return g


def behavioral(img_in, lut_data):
    """
    Fixed-point trilinear 3D LUT interpolation matching the Vitis HLS implementation.

    Simulates ap_ufixed<9,1> (8 frac bits) for LUT values and distance fractions,
    and ap_ufixed<16,6> (10 frac bits) for intermediate accumulation, with truncation
    (floor toward zero) at each quantization step.
    """
    pixels = img_in.reshape(-1, 3).astype(np.int64)
    ch_r = pixels[:, 0]  # channel 0 (bits[7:0])
    ch_g = pixels[:, 1]  # channel 1 (bits[15:8])
    ch_b = pixels[:, 2]  # channel 2 (bits[23:16])

    # Parse LUT: uint32 bit-cast to float32, then to float64
    lut_f32 = lut_data.view(np.float32).astype(np.float64)
    lut_3d_raw = lut_f32.reshape(LUT_DIM, LUT_DIM, LUT_DIM, 3)

    # Quantize LUT entries to ap_ufixed<9,1> (8 frac bits): truncate float to 8 frac bits
    # stmp = *((float*)(&r_int)) then stored as _FIXED_LUT_TYPE
    lut_3d = fp9_1(lut_3d_raw)

    N = len(ch_r)
    MAXVAL = 255  # __MAXVAL(8) for 8-bit input

    # Step 2: scale_i = ch_i / 255.0, quantized to ap_ufixed<9,1>
    # HLS: _FIXED_LUT_TYPE scale_r = (int)inPixR / (float)(__MAXVAL(step));
    scale_r = fp9_1(ch_r.astype(np.float64) / MAXVAL)
    scale_g = fp9_1(ch_g.astype(np.float64) / MAXVAL)
    scale_b = fp9_1(ch_b.astype(np.float64) / MAXVAL)

    # Step 3: index_i = scale_i * (lutdim - 1), quantized to ap_ufixed<16,6>
    # HLS: _FIXED_PIXEL_TYPE index_r = scale_r * (lutdim - 1);
    index_r = fp16_6(scale_r * (LUT_DIM - 1))
    index_g = fp16_6(scale_g * (LUT_DIM - 1))
    index_b = fp16_6(scale_b * (LUT_DIM - 1))

    # Step 4: integer pixel index = int(index.to_float()) -> floor of the 10-frac-bit value
    # HLS: pixelIndex.R = (int)(index_r.to_float());
    pir = np.floor(index_r).astype(np.int32)
    pig = np.floor(index_g).astype(np.int32)
    pib = np.floor(index_b).astype(np.int32)

    # Step 5: distance fraction = index - floor(index), quantized to ap_ufixed<9,1>
    # HLS: _FIXED_LUT_TYPE dist_r = index_r - pixelIndex.R;
    # The subtraction result is a ap_ufixed<16,6> value in [0,1), then stored as ap_ufixed<9,1>
    dist_r = fp9_1(index_r - pir.astype(np.float64))
    dist_g = fp9_1(index_g - pig.astype(np.float64))
    dist_b = fp9_1(index_b - pib.astype(np.float64))

    out = np.zeros((N, 3), dtype=np.float64)

    # Step 6: Border handling (priority order: R-border, G-border, B-border)
    # HLS checks: if (pixelIndex.R + 1 == lutdim) -> R-border lookup [B][G]
    #             elif (pixelIndex.G + 1 == lutdim) -> G-border lookup [B][R]
    #             elif (pixelIndex.B + 1 == lutdim) -> B-border lookup [G][R]
    r_border = (pir + 1 == LUT_DIM)
    g_border = (~r_border) & (pig + 1 == LUT_DIM)
    b_border = (~r_border) & (~g_border) & (pib + 1 == LUT_DIM)
    interior = (~r_border) & (~g_border) & (~b_border)

    # R-border: use borderLutRX[B][G] = lut_3d[B][G][LUT_DIM-1]
    rb_idx = np.where(r_border)[0]
    if len(rb_idx) > 0:
        ib = np.clip(pib[rb_idx], 0, LUT_DIM - 1)
        ig = np.clip(pig[rb_idx], 0, LUT_DIM - 1)
        out[rb_idx] = lut_3d[ib, ig, LUT_DIM - 1]

    # G-border: use borderLutRY[B][R] = lut_3d[B][LUT_DIM-1][R]
    gb_idx = np.where(g_border)[0]
    if len(gb_idx) > 0:
        ib = np.clip(pib[gb_idx], 0, LUT_DIM - 1)
        ir = np.clip(pir[gb_idx], 0, LUT_DIM - 1)
        out[gb_idx] = lut_3d[ib, LUT_DIM - 1, ir]

    # B-border: use borderLutRZ[G][R] = lut_3d[LUT_DIM-1][G][R]
    bb_idx = np.where(b_border)[0]
    if len(bb_idx) > 0:
        ig = np.clip(pig[bb_idx], 0, LUT_DIM - 1)
        ir = np.clip(pir[bb_idx], 0, LUT_DIM - 1)
        out[bb_idx] = lut_3d[LUT_DIM - 1, ig, ir]

    # Step 7: Interior trilinear interpolation
    ni = np.where(interior)[0]
    if len(ni) > 0:
        ir = pir[ni]
        ig = pig[ni]
        ib = pib[ni]
        dr = dist_r[ni]
        dg = dist_g[ni]
        db = dist_b[ni]

        for ch in range(3):
            # 8 cube corners: lut_3d[z=B][y=G][x=R]
            p000 = lut_3d[ib,     ig,     ir,     ch]
            p001 = lut_3d[ib,     ig,     ir + 1, ch]
            p010 = lut_3d[ib,     ig + 1, ir,     ch]
            p011 = lut_3d[ib,     ig + 1, ir + 1, ch]
            p100 = lut_3d[ib + 1, ig,     ir,     ch]
            p101 = lut_3d[ib + 1, ig,     ir + 1, ch]
            p110 = lut_3d[ib + 1, ig + 1, ir,     ch]
            p111 = lut_3d[ib + 1, ig + 1, ir + 1, ch]

            # Trilinear interpolation with fixed-point arithmetic
            # HLS interp3: a=interp1(P000,P100,dr), b=interp1(P001,P101,dr),
            #              c=interp1(P010,P110,dr), d=interp1(P011,P111,dr),
            #              e=interp1(a,b,dg), f=interp1(c,d,dg), g=interp1(e,f,db)
            out[ni, ch] = interp3_fp(p000, p001, p010, p011, p100, p101, p110, p111, dr, dg, db)

    # Step 8: Output conversion
    # HLS: _FIXED_OUT_PIXEL_TYPE _outR = outR * __max;  (__max = 255.0 as ap_ufixed<9,8>)
    # ap_ufixed<9,8> has 1 fractional bit: truncates to 1 frac bit
    # Then cast to uint8 takes integer part (bits [8:1] -> range [0,255])
    # Equivalent: out_uint8 = floor(out_fp10 * 255 * 2) >> 1 = floor(out_fp10 * 255)
    # but with ap_ufixed<9,8> intermediate: floor(out_fp10 * 255 * 2) / 2 cast to int
    # ap_ufixed<9,8>: 1 frac bit -> values are multiples of 0.5
    # (unsigned char)(_outR) takes bits [8:1] which is the integer part of ap_ufixed<9,8>
    out_scaled = np.floor(out * 255.0 * 2.0).astype(np.int64)  # in units of 0.5
    out_bytes = (out_scaled >> 1).astype(np.int32)  # integer part
    out_bytes = np.clip(out_bytes, 0, 255).astype(np.uint8)

    # Flatten interleaved
    return out_bytes.flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
