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
to a {WIDTH}x{HEIGHT} 3-channel 8-bit image using trilinear interpolation.

Parameters:
    img_in:  Pointer to flat array of {HEIGHT * WIDTH * 3} uint8 values representing
             a {HEIGHT}x{WIDTH} image in row-major, channel-interleaved order (ch0,ch1,ch2,...).
    img_out: Pointer to flat array of {HEIGHT * WIDTH * 3} uint8 values for the output image,
             same format as input.
    lut:     Pointer to flat array of {LUT_DIM**3 * 3} uint32 values representing the 3D LUT.
             The LUT is stored as {LUT_DIM}x{LUT_DIM}x{LUT_DIM} entries in [z][y][x] order,
             with 3 float32 values per entry (ch0, ch1, ch2) bit-cast to uint32.
             Each float32 LUT value is in [0.0, 1.0] range.

Algorithm:
    1. For each pixel, extract 3 channels (each 0-255): ch0 from bits[7:0], ch1 from bits[15:8],
       ch2 from bits[23:16].
    2. Compute normalized coordinates: scale_i = channel_i / 255.0
    3. Compute LUT indices: index_i = scale_i * ({LUT_DIM} - 1)
    4. Decompose into integer part (grid cell) and fractional part (distance within cell).
    5. If any index equals {LUT_DIM}-1 (border), use the direct LUT value without interpolation.
    6. Otherwise, trilinear interpolation across the 8 surrounding LUT vertices.
    7. Multiply interpolated [0,1] result by 255 and truncate to uint8.

The trilinear interpolation uses:
    interp1(v1, v2, t) = v1 + t * (v2 - v1)
    Applied successively: first along z-axis with dist_x, then along x-axis with dist_y,
    then along y-axis with dist_z.
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


def behavioral(img_in, lut_data):
    """Pure float trilinear 3D LUT interpolation matching the reference implementation."""
    pixels = img_in.reshape(-1, 3).astype(np.float64)
    ch_r = pixels[:, 0]  # channel 0 (bits[7:0])
    ch_g = pixels[:, 1]  # channel 1 (bits[15:8])
    ch_b = pixels[:, 2]  # channel 2 (bits[23:16])

    # Parse LUT: uint32 bit-cast to float32
    lut_f32 = lut_data.view(np.float32).astype(np.float64)
    lut_3d = lut_f32.reshape(LUT_DIM, LUT_DIM, LUT_DIM, 3)

    N = len(ch_r)

    # Compute normalized coordinates and LUT indices
    scale_r = ch_r / 255.0
    scale_g = ch_g / 255.0
    scale_b = ch_b / 255.0

    index_r = scale_r * (LUT_DIM - 1)
    index_g = scale_g * (LUT_DIM - 1)
    index_b = scale_b * (LUT_DIM - 1)

    idx_r = np.floor(index_r).astype(np.int32)
    idx_g = np.floor(index_g).astype(np.int32)
    idx_b = np.floor(index_b).astype(np.int32)

    dist_r = index_r - idx_r
    dist_g = index_g - idx_g
    dist_b = index_b - idx_b

    out = np.zeros((N, 3), dtype=np.float64)

    # Border: any index at max -> direct lookup, no interpolation
    border = (idx_r == LUT_DIM - 1) | (idx_g == LUT_DIM - 1) | (idx_b == LUT_DIM - 1)
    bi = np.where(border)[0]
    if len(bi) > 0:
        ir_b = np.clip(idx_r[bi], 0, LUT_DIM - 1)
        ig_b = np.clip(idx_g[bi], 0, LUT_DIM - 1)
        ib_b = np.clip(idx_b[bi], 0, LUT_DIM - 1)
        out[bi] = lut_3d[ib_b, ig_b, ir_b]

    # Non-border: trilinear interpolation
    ni = np.where(~border)[0]
    if len(ni) > 0:
        ir = idx_r[ni]
        ig = idx_g[ni]
        ib = idx_b[ni]
        dr = dist_r[ni]
        dg = dist_g[ni]
        db = dist_b[ni]

        for ch in range(3):
            # 8 cube corners: lut[z][y][x]
            p000 = lut_3d[ib, ig, ir, ch]
            p001 = lut_3d[ib, ig, ir + 1, ch]
            p010 = lut_3d[ib, ig + 1, ir, ch]
            p011 = lut_3d[ib, ig + 1, ir + 1, ch]
            p100 = lut_3d[ib + 1, ig, ir, ch]
            p101 = lut_3d[ib + 1, ig, ir + 1, ch]
            p110 = lut_3d[ib + 1, ig + 1, ir, ch]
            p111 = lut_3d[ib + 1, ig + 1, ir + 1, ch]

            # Trilinear: interp along z (B) with dist_r, along x (R) with dist_g, along y (G) with dist_b
            a = p000 + dr * (p100 - p000)
            b = p001 + dr * (p101 - p001)
            c = p010 + dr * (p110 - p010)
            d = p011 + dr * (p111 - p011)

            e = a + dg * (b - a)
            f = c + dg * (d - c)

            g = e + db * (f - e)

            out[ni, ch] = g

    # Convert to uint8: multiply by 255 and truncate
    out_bytes = np.floor(out * 255.0).astype(np.int32)
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
