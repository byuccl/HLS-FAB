import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
NUM_OUT_CH = 3
STATS_SIZE = 256
FINAL_BINS_NUM = 4
MAX_ROWS = 8
MAX_COLS = 8
ZONE_COL_NUM = 8
ZONE_ROW_NUM = 8
NUM_ZONES = ZONE_COL_NUM * ZONE_ROW_NUM  # 64
ROI_TLX = 0
ROI_TLY = 0
ROI_BRX = 127
ROI_BRY = 127
INPUT_MIN = 0.0
INPUT_MAX = 255.0

# Output stats total: num_zones * NUM_OUT_CH * STATS_SIZE
STATS_TOTAL = STATS_SIZE * MAX_ROWS * MAX_COLS * NUM_OUT_CH  # 49152

# Input image: 24 bits per pixel packed into 32-bit words as a bit stream
PIXEL_BITS = 24  # XF_8UC3 = 8 bits * 3 channels
INPUT_PTR_WIDTH = 32
TOTAL_PIXELS = HEIGHT * WIDTH
TOTAL_INPUT_WORDS = (TOTAL_PIXELS * PIXEL_BITS + INPUT_PTR_WIDTH - 1) // INPUT_PTR_WIDTH  # 12288

function_signature = (
    "void ispstats_accel("
    "ap_uint<32>* img_in, "
    "ap_uint<32>* stats, "
    "ap_uint<13>* max_bins)"
)

description = f"""ISP Statistics: computes per-zone histograms of a BGR image.

The input is a {HEIGHT}x{WIDTH} BGR image with 3 channels of 8-bit unsigned values,
packed as a bit stream into {TOTAL_INPUT_WORDS} 32-bit words. Each pixel occupies {PIXEL_BITS} contiguous
bits (B in bits 0-7, G in bits 8-15, R in bits 16-23 of each pixel's 24-bit segment).
Pixels are packed consecutively across word boundaries.

The image is divided into a grid of {ZONE_ROW_NUM} rows x {ZONE_COL_NUM} columns = {NUM_ZONES} zones,
with an ROI covering the full image (top-left ({ROI_TLX},{ROI_TLY}), bottom-right ({ROI_BRX},{ROI_BRY})).
Each zone is {(ROI_BRX - ROI_TLX + 1) // ZONE_COL_NUM} pixels wide and {(ROI_BRY - ROI_TLY + 1) // ZONE_ROW_NUM} pixels tall.

For each pixel in the ROI:
1. Determine zone index: zone_col = (col - {ROI_TLY}) / zone_width, zone_row = (row - {ROI_TLX}) / zone_height,
   zone_idx = zone_row * {ZONE_COL_NUM} + zone_col.
2. For each of the 3 BGR channels, compute histogram bin = int((pixel_value + 0.5) * 1.0) = pixel_value
   (for the default range 0-255 with {STATS_SIZE} bins).
3. Increment stats[zone_idx][channel][bin].

The output stats array has {STATS_TOTAL} uint32 elements, laid out as:
stats[(zone * {STATS_SIZE * NUM_OUT_CH}) + (channel * {STATS_SIZE}) + bin].

The max_bins input is a {FINAL_BINS_NUM}-element array of bin boundaries (unused in this non-merge mode,
but must be provided). Typical values: [63, 127, 191, 255].
"""

inputs_info = [
    {
        "name": "img_in",
        "dtype": "uint32",
        "array_dimension": 1,
        "shape": (TOTAL_INPUT_WORDS,),
        "total_elements": TOTAL_INPUT_WORDS,
        "iotype": "in",
        "min": 0,
        "max": (1 << 32) - 1,
    },
    {
        "name": "stats",
        "dtype": "uint32",
        "array_dimension": 1,
        "shape": (STATS_TOTAL,),
        "total_elements": STATS_TOTAL,
        "iotype": "out",
    },
    {
        "name": "max_bins",
        "dtype": "uint16",
        "array_dimension": 1,
        "shape": (FINAL_BINS_NUM,),
        "total_elements": FINAL_BINS_NUM,
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
]


def _unpack_bitstream(words, num_pixels, pixel_bits, ptr_width):
    """Unpack a bit-stream of pixel_bits-wide pixels from ptr_width-wide words."""
    # Convert words to a single large bit buffer
    words = words.astype(np.uint64)
    pixels = np.zeros(num_pixels, dtype=np.uint32)
    bit_pos = 0
    word_idx = 0
    remaining_bits = 0
    buf = np.uint64(0)

    for p in range(num_pixels):
        # Ensure we have enough bits
        while remaining_bits < pixel_bits:
            if word_idx < len(words):
                buf |= np.uint64(words[word_idx]) << np.uint64(remaining_bits)
                word_idx += 1
                remaining_bits += ptr_width
            else:
                break
        pixels[p] = np.uint32(buf & np.uint64((1 << pixel_bits) - 1))
        buf >>= np.uint64(pixel_bits)
        remaining_bits -= pixel_bits

    return pixels


def behavioral(img_in, max_bins_in):
    img_in = img_in.astype(np.uint32)

    # Unpack bit-stream: 24-bit pixels from 32-bit words
    pixel_data = _unpack_bitstream(img_in, TOTAL_PIXELS, PIXEL_BITS, INPUT_PTR_WIDTH)
    pixel_data = pixel_data.reshape(HEIGHT, WIDTH)

    # Extract BGR channels from 24-bit pixels
    b_ch = (pixel_data & 0xFF).astype(np.uint8)
    g_ch = ((pixel_data >> 8) & 0xFF).astype(np.uint8)
    r_ch = ((pixel_data >> 16) & 0xFF).astype(np.uint8)
    channels = [b_ch, g_ch, r_ch]

    zone_width = (ROI_BRX - ROI_TLX + 1) // ZONE_COL_NUM   # 16
    zone_height = (ROI_BRY - ROI_TLY + 1) // ZONE_ROW_NUM  # 16

    # Compute bin mapping: for default params, bin = pixel_value
    min_val = INPUT_MIN - 0.5
    max_val = INPUT_MAX + 0.5
    interval = (max_val - min_val) / STATS_SIZE
    internal_inv = 1.0 / interval

    # Initialize stats array: [NUM_ZONES][NUM_OUT_CH][STATS_SIZE]
    stats_array = np.zeros((NUM_ZONES, NUM_OUT_CH, STATS_SIZE), dtype=np.uint32)

    for row in range(HEIGHT):
        for col in range(WIDTH):
            if (row >= ROI_TLX and row <= ROI_BRX and
                col >= ROI_TLY and col <= ROI_BRY):
                zone_col = (col - ROI_TLY) // zone_width
                zone_row = (row - ROI_TLX) // zone_height
                zone_idx = zone_row * ZONE_COL_NUM + zone_col

                for ch in range(NUM_OUT_CH):
                    val = channels[ch][row, col]
                    bin_idx = int((val - min_val) * internal_inv)
                    if bin_idx < 0:
                        bin_idx = 0
                    if bin_idx >= STATS_SIZE:
                        bin_idx = STATS_SIZE - 1
                    stats_array[zone_idx, ch, bin_idx] += 1

    # Flatten to output layout: stats[(zone * STATS_SIZE * NUM_OUT_CH) + (ch * STATS_SIZE) + bin]
    num_final_bins = STATS_SIZE  # MERGE_BINS=0
    total_bins = num_final_bins * NUM_OUT_CH
    stats_out = np.zeros(STATS_TOTAL, dtype=np.uint32)
    for k in range(NUM_ZONES):
        for ch in range(NUM_OUT_CH):
            for i in range(STATS_SIZE):
                stats_out[k * total_bins + ch * num_final_bins + i] = stats_array[k, ch, i]

    return stats_out.astype(np.uint32).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
