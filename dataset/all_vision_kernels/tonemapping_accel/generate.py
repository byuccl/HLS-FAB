import numpy as np
from hlsfab import PromptConstructor, generate_examples
import struct

HEIGHT = 512
WIDTH = 512
BLOCK_HEIGHT = 64  # compile-time minimum
BLOCK_WIDTH = 64   # compile-time minimum
BLK_HEIGHT_RT = 128  # runtime block height from testbench
BLK_WIDTH_RT = 128   # runtime block width from testbench
NPC = 4
NUM_CHANNELS = 3
PIXEL_BITS = 24  # 8 bits * 3 channels
INPUT_PTR_WIDTH = 128
OUTPUT_PTR_WIDTH = 128
OUT_DEPTH = 255.0

TOTAL_PIXELS = HEIGHT * WIDTH
TOTAL_INPUT_WORDS = (TOTAL_PIXELS * PIXEL_BITS + INPUT_PTR_WIDTH - 1) // INPUT_PTR_WIDTH
TOTAL_OUTPUT_WORDS = TOTAL_INPUT_WORDS  # same type

function_signature = (
    "void tonemapping_accel("
    "ap_uint<128>* in_ptr, "
    "ap_uint<128>* out_ptr)"
)

description = f"""Local Tone Mapping (LTM): applies adaptive local tone mapping to a {HEIGHT}x{WIDTH} BGR image.

The input is a {HEIGHT}x{WIDTH} BGR image with 3 channels of 8-bit unsigned values,
packed as a bit stream into {TOTAL_INPUT_WORDS} 128-bit words. Each pixel occupies {PIXEL_BITS} contiguous
bits (B in bits 0-7, G in bits 8-15, R in bits 16-23 of each pixel's 24-bit segment).
Pixels are packed consecutively across word boundaries.

The output is a {HEIGHT}x{WIDTH} BGR image in the same packed format.

The algorithm uses a two-pass approach:
- Pass 1 (warmup): processes the image to compute per-block min/max of log-luminance.
  The output of this pass is discarded.
- Pass 2 (actual): uses the min/max from pass 1 to apply local tone mapping.

For each pixel, the maximum channel value across BGR determines the luminance.
Log-luminance = log(max_channel + epsilon) where epsilon = ~1.1920929e-07 (float epsilon).

The image is divided into blocks of {BLK_HEIGHT_RT}x{BLK_WIDTH_RT} pixels. Per-block min/max
of log-luminance are computed, then bilinearly interpolated across the image to get
smooth per-pixel min/max estimates.

For each output pixel:
1. max_log = max(log(B+eps), log(G+eps), log(R+eps))
2. max_in = max(B, G, R) (integer max)
3. compute_val = (max_log - interp_min) / (interp_max - interp_min + eps)
4. ratio = compute_val / (max_in + eps)
5. For each channel: out = clamp(channel_val * ratio * 255, 0, 255)

The output is uint8 per channel, clamped to [0, 255].
"""

inputs_info = [
    {
        "name": "in_ptr",
        "dtype": "uint32",
        "array_dimension": 1,
        "shape": (TOTAL_INPUT_WORDS * 4,),  # 128-bit words stored as 4 uint32s each
        "total_elements": TOTAL_INPUT_WORDS * 4,
        "iotype": "in",
        "min": 0,
        "max": (1 << 32) - 1,
    },
    {
        "name": "out_ptr",
        "dtype": "uint32",
        "array_dimension": 1,
        "shape": (TOTAL_OUTPUT_WORDS * 4,),
        "total_elements": TOTAL_OUTPUT_WORDS * 4,
        "iotype": "out",
    },
]

EPS = np.float32(np.finfo(np.float32).eps)  # ~1.1920929e-07


def _unpack_bitstream(words_u32, num_pixels, pixel_bits, ptr_width_bits):
    """Unpack a bit-stream of pixel_bits-wide pixels from ptr_width_bits-wide words.
    words_u32: flat array of uint32, representing packed ptr_width_bits-wide words in little-endian.
    """
    # Reconstruct byte stream
    byte_data = words_u32.astype(np.uint32).tobytes()
    total_bits = num_pixels * pixel_bits
    pixels = np.zeros(num_pixels, dtype=np.uint32)

    buf = 0
    remaining = 0
    byte_idx = 0

    for p in range(num_pixels):
        while remaining < pixel_bits:
            if byte_idx < len(byte_data):
                buf |= int(byte_data[byte_idx]) << remaining
                byte_idx += 1
                remaining += 8
            else:
                break
        pixels[p] = buf & ((1 << pixel_bits) - 1)
        buf >>= pixel_bits
        remaining -= pixel_bits

    return pixels


def _pack_bitstream(pixels, num_pixels, pixel_bits, total_u32):
    """Pack pixel_bits-wide pixels into a flat uint32 array (little-endian byte stream)."""
    total_bytes = total_u32 * 4
    byte_arr = bytearray(total_bytes)

    buf = 0
    remaining = 0
    byte_idx = 0

    for p in range(num_pixels):
        buf |= int(pixels[p]) << remaining
        remaining += pixel_bits
        while remaining >= 8:
            if byte_idx < total_bytes:
                byte_arr[byte_idx] = buf & 0xFF
                buf >>= 8
                remaining -= 8
                byte_idx += 1

    if remaining > 0 and byte_idx < total_bytes:
        byte_arr[byte_idx] = buf & 0xFF

    return np.frombuffer(bytes(byte_arr), dtype=np.uint32)


class TileInfo:
    """Replicate LTMTile logic for runtime block sizes."""
    def __init__(self, rows, cols, blk_rows, blk_cols, npc):
        self.rows = rows
        self.cols = cols
        self.blk_rows = blk_rows
        self.blk_cols = blk_cols
        self.npc = npc
        self.npc_shift = int(np.log2(npc))

        self.cols_npc = (cols + npc - 1) >> self.npc_shift

        # Find blkColsLog2
        blk_cols_log2_min = int(np.log2(BLOCK_WIDTH))  # compile-time minimum
        blk_cols_log2 = blk_cols_log2_min
        cols_log2_max = int(np.log2(cols)) if cols > 0 else 0
        while blk_cols_log2 <= cols_log2_max:
            if (1 << blk_cols_log2) >= blk_cols:
                break
            blk_cols_log2 += 1
        self.blk_cols_log2 = blk_cols_log2
        self.blk_cols_npc = 1 << (blk_cols_log2 - self.npc_shift)

        self.h_blk_count = (self.cols_npc + self.blk_cols_npc - 1) >> blk_cols_log2
        th = self.h_blk_count << blk_cols_log2
        self.h_blk_size = [self.blk_cols_npc, self.blk_cols_npc]
        if self.cols_npc < th:
            t = self.cols_npc - ((self.cols_npc >> blk_cols_log2) << blk_cols_log2)
            self.h_blk_size[0] = t >> 1
            self.h_blk_count += 1

        # Vertical
        blk_rows_log2_min = int(np.log2(BLOCK_HEIGHT))
        blk_rows_log2 = blk_rows_log2_min
        rows_log2_max = int(np.log2(rows)) if rows > 0 else 0
        while blk_rows_log2 <= rows_log2_max:
            if (1 << blk_rows_log2) >= blk_rows:
                break
            blk_rows_log2 += 1
        self.blk_rows_log2 = blk_rows_log2

        self.v_blk_count = (rows + blk_rows - 1) >> blk_rows_log2
        tv = self.v_blk_count << blk_rows_log2
        self.v_blk_size = [blk_rows, blk_rows]
        if rows < tv:
            t = rows - ((rows >> blk_rows_log2) << blk_rows_log2)
            self.v_blk_size[0] = t >> 1
            self.v_blk_count += 1

        # MinMax array sizes (compile-time)
        BLK_COLS_NPC_ALIGNED = BLOCK_WIDTH >> self.npc_shift
        COLS_NPC_ALIGNED_CT = (WIDTH + npc - 1) >> self.npc_shift
        if (HEIGHT % BLOCK_HEIGHT) == 0:
            self.mm_v_size = (HEIGHT // BLOCK_HEIGHT) + 1
        else:
            self.mm_v_size = ((HEIGHT + BLOCK_HEIGHT - 1) // BLOCK_HEIGHT + 1) + 1
        if (COLS_NPC_ALIGNED_CT % BLK_COLS_NPC_ALIGNED) == 0:
            self.mm_h_size = (COLS_NPC_ALIGNED_CT // BLK_COLS_NPC_ALIGNED) + 1
        else:
            self.mm_h_size = ((COLS_NPC_ALIGNED_CT + BLK_COLS_NPC_ALIGNED - 1) // BLK_COLS_NPC_ALIGNED + 1) + 1

    def get_v_blk_size(self, index):
        if index == 0 or index == self.v_blk_count - 1:
            return self.v_blk_size[0]
        return self.v_blk_size[1]

    def get_minmax_v_blk_size(self, index):
        if index == 0:
            return self.v_blk_size[0] >> 1
        if index == self.v_blk_count:
            return self.v_blk_size[0] >> 1
        if index == 1:
            return (self.v_blk_size[0] + self.v_blk_size[1]) >> 1
        if index == self.v_blk_count - 1:
            return (self.v_blk_size[0] + self.v_blk_size[1]) >> 1
        if index < self.v_blk_count - 1:
            return self.v_blk_size[1]
        return 0

    def get_h_blk_size(self, j_npc):
        """j_npc is in NPC-aligned columns."""
        # Convert to pixel column to check boundary
        if j_npc < self.h_blk_size[0]:
            return self.h_blk_size[0]
        if j_npc >= (self.cols_npc - self.h_blk_size[0]):
            return self.h_blk_size[0]
        return self.h_blk_size[1]


def _float_to_raw_bits(f):
    return struct.unpack('<I', struct.pack('<f', np.float32(f)))[0]


def _raw_bits_to_float(u):
    return struct.unpack('<f', struct.pack('<I', int(u) & 0xFFFFFFFF))[0]


def _ltm_single_pass(pixels_bgr, tile, omin_r, omax_r):
    """Run one pass of LTM. Returns (output_pixels, omin_w, omax_w)."""
    H, W = tile.rows, tile.cols

    # Step 1: compute log image and per-pixel max of logs
    b = pixels_bgr[:, :, 0].astype(np.float32)
    g = pixels_bgr[:, :, 1].astype(np.float32)
    r = pixels_bgr[:, :, 2].astype(np.float32)

    log_b = np.log(b + EPS)
    log_g = np.log(g + EPS)
    log_r = np.log(r + EPS)

    max_log = np.maximum(np.maximum(log_b, log_g), log_r)
    max_in = np.maximum(np.maximum(pixels_bgr[:, :, 0].astype(np.float32),
                                    pixels_bgr[:, :, 1].astype(np.float32)),
                         pixels_bgr[:, :, 2].astype(np.float32))

    # Step 2: compute block min/max from max_log
    # The min/max computation operates on NPC-aligned columns in groups
    omin_w = np.full((tile.mm_v_size, tile.mm_h_size), _float_to_raw_bits(np.float32(np.log(float(0xFFFFFFFF)))), dtype=np.uint32)
    omax_w = np.full((tile.mm_v_size, tile.mm_h_size), 0, dtype=np.uint32)

    # Process min/max per vertical block boundary
    for vi in range(tile.v_blk_count + 1):
        v_blk_size = tile.get_minmax_v_blk_size(vi)
        if v_blk_size == 0:
            continue

        # Determine start row for this minmax block
        if vi == 0:
            row_start = 0
        elif vi == 1:
            row_start = tile.v_blk_size[0] >> 1
        else:
            row_start = (tile.v_blk_size[0] >> 1) + (tile.v_blk_size[0] + tile.v_blk_size[1]) // 2
            if vi > 2:
                row_start = (tile.v_blk_size[0] >> 1) + ((tile.v_blk_size[0] + tile.v_blk_size[1]) >> 1)
                for k in range(2, vi):
                    if k < tile.v_blk_count - 1:
                        row_start += tile.v_blk_size[1]
                    elif k == tile.v_blk_count - 1:
                        row_start += (tile.v_blk_size[0] + tile.v_blk_size[1]) >> 1

        row_end = min(row_start + v_blk_size, H)
        if row_start >= H:
            continue

        omin_l = [_float_to_raw_bits(np.float32(np.log(float(0xFFFFFFFF))))] * tile.mm_h_size
        omax_l = [0] * tile.mm_h_size

        b_start = True
        for row_off in range(row_end - row_start):
            row = row_start + row_off
            if row >= H:
                break
            blk = 0
            offset = 0
            for j in range(tile.cols_npc):
                h_blk_size = tile.get_h_blk_size(j)

                # Get min/max across NPC pixels at this position
                pin_val_min_bits = _float_to_raw_bits(np.float32(np.log(float(0xFFFFFFFF))))
                pin_val_max_bits = 0

                for l in range(NPC):
                    col = j * NPC + l
                    if col >= W:
                        break
                    val = max_log[row, col]
                    val_bits = _float_to_raw_bits(val)

                    if _raw_bits_to_float(val_bits) < _raw_bits_to_float(pin_val_min_bits):
                        pin_val_min_bits = val_bits
                    if _raw_bits_to_float(val_bits) > _raw_bits_to_float(pin_val_max_bits):
                        pin_val_max_bits = val_bits

                cur_min = omin_l[blk]
                cur_max = omax_l[blk]
                if b_start:
                    cur_min = pin_val_min_bits
                    cur_max = pin_val_min_bits  # Note: HLS code sets max=pinValMin on first
                else:
                    if _raw_bits_to_float(pin_val_min_bits) < _raw_bits_to_float(cur_min):
                        cur_min = pin_val_min_bits
                    if _raw_bits_to_float(pin_val_max_bits) > _raw_bits_to_float(cur_max):
                        cur_max = pin_val_max_bits

                offset += 1
                b_start = False
                if offset == (h_blk_size >> 1):
                    blk += 1
                    b_start = (row_off == 0)
                if offset == h_blk_size:
                    offset = 0

                omin_l[blk] = cur_min
                omax_l[blk] = cur_max
                omin_w[vi][blk] = cur_min
                omax_w[vi][blk] = cur_max

    # Step 3: bilinear interpolation of omin_r and omax_r
    interp_min = np.zeros((H, W), dtype=np.float32)
    interp_max = np.zeros((H, W), dtype=np.float32)

    for arr_r, interp_out in [(omin_r, interp_min), (omax_r, interp_max)]:
        row_idx = 0
        for vi in range(tile.v_blk_count):
            v_blk_size = tile.get_v_blk_size(vi)
            for i in range(v_blk_size):
                if row_idx >= H:
                    break
                blk = 0
                offset = 0
                for j in range(tile.cols_npc):
                    h_blk_size = tile.get_h_blk_size(j)

                    a = [0.0] * 4
                    for m in range(2):
                        for n in range(2):
                            idx_v = vi + m
                            idx_h = blk + n
                            if idx_v < tile.mm_v_size and idx_h < tile.mm_h_size:
                                a[m * 2 + n] = _raw_bits_to_float(arr_r[idx_v][idx_h])
                            else:
                                a[m * 2 + n] = 0.0

                    for l in range(NPC):
                        col = j * NPC + l
                        if col >= W:
                            break
                        # Bilinear interpolation
                        x1 = a[0] + (a[1] - a[0]) * offset / h_blk_size
                        x2 = a[2] + (a[3] - a[2]) * offset / h_blk_size
                        y = x1 + (x2 - x1) * i / v_blk_size
                        interp_out[row_idx, col] = y

                    offset += 1
                    if offset == h_blk_size:
                        offset = 0
                        blk += 1

                row_idx += 1

    # Step 4: compute output
    out_pixels = np.zeros((H, W, 3), dtype=np.uint8)
    for row in range(H):
        for col in range(W):
            ml = max_log[row, col]
            mi = max_in[row, col]
            i_min = interp_min[row, col]
            i_max = interp_max[row, col]

            compute_val = (ml - i_min) / (i_max - i_min + EPS)
            ratio = compute_val / (mi + EPS)

            for ch in range(3):
                val = float(pixels_bgr[row, col, ch])
                final_val = val * ratio * OUT_DEPTH
                if final_val < 0:
                    out_pixels[row, col, ch] = 0
                elif final_val > 255:
                    out_pixels[row, col, ch] = 255
                else:
                    out_pixels[row, col, ch] = int(final_val)

    return out_pixels, omin_w, omax_w


def behavioral(in_ptr):
    in_ptr = in_ptr.astype(np.uint32)

    # Unpack bit-stream
    pixel_data = _unpack_bitstream(in_ptr, TOTAL_PIXELS, PIXEL_BITS, INPUT_PTR_WIDTH)

    # Extract BGR
    b = (pixel_data & 0xFF).astype(np.uint8)
    g = ((pixel_data >> 8) & 0xFF).astype(np.uint8)
    r = ((pixel_data >> 16) & 0xFF).astype(np.uint8)

    pixels_bgr = np.stack([b.reshape(HEIGHT, WIDTH),
                           g.reshape(HEIGHT, WIDTH),
                           r.reshape(HEIGHT, WIDTH)], axis=2)

    tile = TileInfo(HEIGHT, WIDTH, BLK_HEIGHT_RT, BLK_WIDTH_RT, NPC)

    # Initialize omin/omax to zeros (simulating static initialization)
    omin_init = np.zeros((tile.mm_v_size, tile.mm_h_size), dtype=np.uint32)
    omax_init = np.zeros((tile.mm_v_size, tile.mm_h_size), dtype=np.uint32)

    # Pass 1: warmup (reads from zeros, writes new min/max)
    _, omin_1, omax_1 = _ltm_single_pass(pixels_bgr, tile, omin_init, omax_init, )

    # Pass 2: actual (reads from pass 1's min/max)
    out_pixels, _, _ = _ltm_single_pass(pixels_bgr, tile, omin_1, omax_1)

    # Pack output back into bit-stream
    out_packed = (out_pixels[:, :, 0].astype(np.uint32).flatten() |
                  (out_pixels[:, :, 1].astype(np.uint32).flatten() << 8) |
                  (out_pixels[:, :, 2].astype(np.uint32).flatten() << 16))

    result = _pack_bitstream(out_packed, TOTAL_PIXELS, PIXEL_BITS, TOTAL_OUTPUT_WORDS * 4)
    return result.astype(np.uint32).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
