import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 1080
WIDTH = 1920

function_signature = "void cvtcolor_rgba2iyuv(int32_t* imgInput, int32_t* imgOutput0, int32_t* imgOutput1, int32_t* imgOutput2)"

description = f"""Converts an RGBA image of size {HEIGHT}x{WIDTH} to IYUV (YUV 4:2:0 planar) format.

The input is a flat array of {HEIGHT*WIDTH} 32-bit packed RGBA pixels, where each pixel packs
R in bits [7:0], G in bits [15:8], B in bits [23:16], and A in bits [31:24].

The function produces three separate output planes:
- imgOutput0: Y (luma) plane of size {HEIGHT}x{WIDTH} (one uint8 per pixel).
- imgOutput1: U (chroma-blue) plane with {HEIGHT*WIDTH//4} uint8 values.
- imgOutput2: V (chroma-red) plane with {HEIGHT*WIDTH//4} uint8 values.

Conversion uses fixed-point Q1.15 arithmetic with the following integer coefficients and
rounding offset F_05=16384 (0.5 in Q1.15):
  Y = ((8422*R + 16516*G + 3212*B + 16384) >> 15) + 16
  U = ((-4850*R + -9536*G + 14386*B + 16384) >> 15) + 128
  V = ((14382*R + -12059*G + -2327*B + 16384) >> 15) + 128

The signed coefficients are interpreted as int16: R2U=60686=-4850, G2U=56000=-9536,
G2V=53477=-12059, B2V=63209=-2327. Results are saturated to [0, 255].

Y is computed for every pixel. U and V are subsampled: only the even-row, even-column
pixels (i.e., the top-left of each 2x2 block) contribute a U/V sample. The result is
{HEIGHT//2} x {WIDTH//2} = {HEIGHT*WIDTH//4} UV samples each, stored in row-major order."""

inputs_info = [
    {
        "name": "imgInput",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": int(np.iinfo(np.uint32).max),
    },
    {
        "name": "imgOutput0",
        "array_dimension": 1,
        "shape": (HEIGHT * WIDTH,),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "imgOutput1",
        "array_dimension": 1,
        "shape": ((HEIGHT // 4) * WIDTH,),
        "total_elements": (HEIGHT // 4) * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "imgOutput2",
        "array_dimension": 1,
        "shape": ((HEIGHT // 4) * WIDTH,),
        "total_elements": (HEIGHT // 4) * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]


def behavioral(imgInput):
    # Exact fixed-point Q1.15 arithmetic matching Vitis KernRgba2Iyuv (NPC=1)
    # Coefficients from xf_cvt_color_utils.hpp (interpreted as signed int16):
    #   R2Y=8422, G2Y=16516, B2Y=3212
    #   R2U=60686->-4850, G2U=56000->-9536, B2U=14386
    #   R2V=14382, G2V=53477->-12059, B2V=63209->-2327
    #   F_05=16384 (rounding offset)
    # saturate_cast: (value >> 15) + offset, clamped to [0,255]
    H, W = HEIGHT, WIDTH

    rgba = imgInput.astype(np.int32)
    R = (rgba & 0xFF)
    G = ((rgba >> 8) & 0xFF)
    B = ((rgba >> 16) & 0xFF)

    # Signed int16 coefficient interpretation
    R2Y, G2Y, B2Y = np.int32(8422), np.int32(16516), np.int32(3212)
    R2U, G2U, B2U = np.int32(-4850), np.int32(-9536), np.int32(14386)
    R2V, G2V, B2V = np.int32(14382), np.int32(-12059), np.int32(-2327)
    F_05 = np.int32(16384)

    # Compute Y for all pixels
    Y_raw = R2Y * R + G2Y * G + B2Y * B + F_05
    Y_shifted = (Y_raw >> 15) + 16
    Y = np.clip(Y_shifted, 0, 255).astype(np.uint8)

    # Compute U and V only at even-row, even-column positions (top-left of each 2x2 block)
    # Reshape to (H, W) to select even rows and even columns
    R2d = R.reshape(H, W)
    G2d = G.reshape(H, W)
    B2d = B.reshape(H, W)

    # Even rows, even columns (U/V subsampled pixels)
    Re = R2d[0::2, 0::2]  # shape (H/2, W/2)
    Ge = G2d[0::2, 0::2]
    Be = B2d[0::2, 0::2]

    U_raw = R2U * Re + G2U * Ge + B2U * Be + F_05
    V_raw = R2V * Re + G2V * Ge + B2V * Be + F_05

    U_vals = np.clip((U_raw >> 15) + 128, 0, 255).astype(np.uint8)
    V_vals = np.clip((V_raw >> 15) + 128, 0, 255).astype(np.uint8)

    # U_vals and V_vals are shape (H/2, W/2), flatten to (H*W/4,)
    U_out = U_vals.flatten()
    V_out = V_vals.flatten()

    return (Y, U_out, V_out)


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
