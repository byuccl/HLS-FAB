import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 676
WIDTH = 1024
TOTAL_PIXELS = HEIGHT * WIDTH
# 3 channels * 8 bits = 24 bits per pixel, packed into 32-bit words
BYTES_PER_PIXEL = 3
TOTAL_BYTES = TOTAL_PIXELS * BYTES_PER_PIXEL
# Number of 32-bit words needed
TOTAL_WORDS = (TOTAL_BYTES + 3) // 4  # ceiling division

# bt2020_bt709 color correction matrix coefficients (hardcoded)
CCM_MATRIX = [
    [1.6605, -0.5876, -0.0728],
    [-0.1246, 1.1329, -0.0083],
    [-0.0182, -0.1006, 1.1187],
]
CCM_OFFSET = [0.0, 0.0, 0.0]

function_signature = "void ccm_accel(int32_t* src, int32_t* dst)"

description = f"""Color correction matrix (CCM) applied to a {HEIGHT}x{WIDTH} 3-channel 8-bit image
using the BT.2020 to BT.709 color space conversion.

The image pixels are stored as a tightly packed byte stream in 32-bit words.
Each pixel occupies 3 consecutive bytes (R, G, B) with no padding between pixels.
The {TOTAL_PIXELS} pixels are packed into {TOTAL_WORDS} 32-bit words.

A fixed 3x3 color correction matrix is applied to each pixel:
  M = [[1.6605, -0.5876, -0.0728],
       [-0.1246,  1.1329, -0.0083],
       [-0.0182, -0.1006,  1.1187]]

For each pixel:
1. Extract R, G, B channels (each uint8) from the packed byte stream.
2. Compute each output channel:
   out_R = R * M[0][0] + G * M[0][1] + B * M[0][2]
   out_G = R * M[1][0] + G * M[1][1] + B * M[1][2]
   out_B = R * M[2][0] + G * M[2][1] + B * M[2][2]
3. The matrix coefficients are represented internally as fixed-point values.
   The float coefficients are pre-scaled by 2^20 to signed integers, then recovered
   by dividing by 2^20 into a fixed-point format with 28 fractional bits.
   Each product (uint8 * coeff) is truncated to 8 fractional bits (floor toward
   negative infinity).
4. The sum is truncated to integer via floor toward negative infinity.
5. Each output channel is saturated to [0, 255].

Parameters:
- src: {TOTAL_WORDS} packed 32-bit words containing {TOTAL_PIXELS} RGB pixels (tightly packed bytes).
- dst: {TOTAL_WORDS} packed 32-bit words for output, same format.
"""

inputs_info = [
    {
        "name": "src",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "int32",
        "shape": (TOTAL_WORDS,),
        "total_elements": TOTAL_WORDS,
        "min": 0,
        "max": 2**32 - 1,
    },
    {
        "name": "dst",
        "iotype": "out",
        "array_dimension": 1,
        "dtype": "int32",
        "shape": (TOTAL_WORDS,),
        "total_elements": TOTAL_WORDS,
        "min": 0,
        "max": 2**32 - 1,
    },
]


def _unpack_rgb_from_words(words, num_pixels):
    """Unpack tightly packed RGB bytes from 32-bit words."""
    # Convert uint32 words to bytes (little-endian)
    byte_data = np.zeros(len(words) * 4, dtype=np.uint8)
    for b in range(4):
        byte_data[b::4] = ((words >> (b * 8)) & 0xFF).astype(np.uint8)
    # Extract R, G, B channels
    r = byte_data[0:num_pixels * 3:3].copy()
    g = byte_data[1:num_pixels * 3:3].copy()
    b = byte_data[2:num_pixels * 3:3].copy()
    return r, g, b


def _pack_rgb_to_words(r, g, b, num_words):
    """Pack RGB bytes tightly into 32-bit words."""
    num_pixels = len(r)
    byte_data = np.zeros(num_words * 4, dtype=np.uint8)
    byte_data[0:num_pixels * 3:3] = r
    byte_data[1:num_pixels * 3:3] = g
    byte_data[2:num_pixels * 3:3] = b
    # Convert bytes to uint32 words (little-endian)
    words = (byte_data[0::4].astype(np.uint32) |
             (byte_data[1::4].astype(np.uint32) << 8) |
             (byte_data[2::4].astype(np.uint32) << 16) |
             (byte_data[3::4].astype(np.uint32) << 24))
    return words


def behavioral(src):
    src = src.astype(np.uint32)

    # Unpack pixels from tightly packed byte stream
    r, g, b_ch = _unpack_rgb_from_words(src, TOTAL_PIXELS)
    r = r.astype(np.float64)
    g = g.astype(np.float64)
    b_ch = b_ch.astype(np.float64)

    # Compute fixed-point matrix coefficients
    # float -> int(float * 2^20) -> divide by 2^20
    mat = np.zeros((3, 3), dtype=np.float64)
    off = np.zeros(3, dtype=np.float64)
    for i in range(3):
        for j in range(3):
            scaled_int = int(np.float32(CCM_MATRIX[i][j]) * np.float32(1048576.0))
            mat[i, j] = scaled_int / 1048576.0
        scaled_off = int(np.float32(CCM_OFFSET[i]) * np.float32(1048576.0))
        off[i] = scaled_off / 1048576.0

    out_r = np.zeros(TOTAL_PIXELS, dtype=np.uint8)
    out_g = np.zeros(TOTAL_PIXELS, dtype=np.uint8)
    out_b = np.zeros(TOTAL_PIXELS, dtype=np.uint8)

    channels_in = [r, g, b_ch]
    channels_out = [out_r, out_g, out_b]

    for ch in range(3):
        # Each product: pixel * coeff -> truncate to 8 frac bits (AP_TRN = floor)
        prod_r = np.floor(channels_in[0] * mat[ch, 0] * 256.0) / 256.0
        prod_g = np.floor(channels_in[1] * mat[ch, 1] * 256.0) / 256.0
        prod_b = np.floor(channels_in[2] * mat[ch, 2] * 256.0) / 256.0

        # Sum + offset, then truncate to int (AP_TRN on ap_fixed -> floor)
        total = prod_r + prod_g + prod_b + off[ch]
        channel = np.floor(total).astype(np.int32)

        # Saturate to [0, 255]
        channel = np.clip(channel, 0, 255)
        channels_out[ch][:] = channel.astype(np.uint8)

    # Pack output back into 32-bit words
    result = _pack_rgb_to_words(out_r, out_g, out_b, TOTAL_WORDS)
    return result.astype(np.uint32).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
