import numpy as np
from hlsfab import PromptConstructor, generate_examples
import struct

HEIGHT = 512
WIDTH = 512
BLOCK_HEIGHT = 64  # compile-time minimum (BLK_ROWS template param)
BLOCK_WIDTH = 64   # compile-time minimum (BLK_COLS template param)
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

The algorithm uses a two-pass approach with static omin/omax arrays:
- Pass 1 (warmup): reads omin/omax initialized to zeros, writes new per-block min/max of
  log-luminance. The output pixels of this pass are discarded.
- Pass 2 (actual): reads the min/max written by pass 1, applies local tone mapping,
  writes output pixels.

The static omin/omax arrays store float values as raw IEEE 754 uint32 bit patterns.

For each pixel, the log of each channel is computed:
  log_ch = log(channel_val + float_epsilon)  where float_epsilon = ~1.1920929e-07
The maximum log across all 3 channels determines the luminance for min/max tracking.

The image is divided into blocks using LTMTile with compile-time BLK_ROWS={BLOCK_HEIGHT},
BLK_COLS={BLOCK_WIDTH} and runtime blk_height={BLK_HEIGHT_RT}, blk_width={BLK_WIDTH_RT}.

LTMTile construction (runtime variant):
  blkColsLog2: smallest k >= log2(BLK_COLS={BLOCK_WIDTH}) such that (1<<k) >= blk_width
  mBlkColsNPCAligned = 1 << (blkColsLog2 - log2(NPC={NPC}))  [NPC-aligned block width]
  mHBlkCount: number of horizontal min/max blocks (covers mColsNPCAligned)
  mHBlkSize[0] = boundary block size (NPC-aligned), mHBlkSize[1] = interior block size
  Similarly for vertical: mVBlkCount, mVBlkSize[0/1]
  If image dimension is exactly divisible by block size: mVBlkSize[0] = mVBlkSize[1] = blk_size
  Otherwise: mVBlkSize[0] = remainder/2 (half remainder at each edge), mVBlkCount incremented

getHBlkSize(addr): addr is NPC-aligned column index (0..mColsNPCAligned-1)
  returns mHBlkSize[0] if addr < mHBlkSize[0] or addr >= (mColsNPCAligned - mHBlkSize[0])
  else returns mHBlkSize[1]

getMinMaxVBlkSize(index):
  index=0:              mVBlkSize[0] >> 1
  index=mVBlkCount:     mVBlkSize[0] >> 1
  index=1:              (mVBlkSize[0] + mVBlkSize[1]) >> 1
  index=mVBlkCount-1:   (mVBlkSize[0] + mVBlkSize[1]) >> 1
  otherwise:            mVBlkSize[1]

Min/Max computation (processMinMax / getMinMax):
  For each vBlkIndex from 0 to mVBlkCount (inclusive), process getMinMaxVBlkSize(i) rows.
  The omin_l/omax_l local arrays persist across vBlkIndex calls (not re-initialized per call).
  bStart resets to True at the start of each getMinMax call.
  For each NPC-aligned column j in each row:
    hBlkSize = getHBlkSize(j)
    pinValMin = min of log-max values across NPC pixels at this column group
    pinValMax = max of log-max values across NPC pixels at this column group
    (min/max comparisons done on raw float bit patterns via rawBitsToFloat)
    If bStart: omin_l[blk] = pinValMin, omax_l[blk] = pinValMin (max also initialized to min!)
    Else: omin_l[blk] = min(omin_l[blk], pinValMin), omax_l[blk] = max(omax_l[blk], pinValMax)
    offset++; bStart = False
    If offset == hBlkSize >> 1: blk++; bStart = (row_in_vblk == 0)
    If offset == hBlkSize: offset = 0
    omin[vBlkIndex][blk] = omin_l[blk]
    omax[vBlkIndex][blk] = omax_l[blk]

Bilinear interpolation of min/max (processInterpolate / interpolate):
  For each vBlkIndex from 0 to mVBlkCount-1, process getVBlkSize(i) rows.
  For each NPC-aligned column j in each row:
    hBlkSize = getHBlkSize(j)
    For each NPC pixel l at this column group:
      a[0..3] = rawBitsToFloat(arr[vBlkIndex+m][blk+n]) for m,n in {0,1}x{0,1}
      x1 = a[0] + (a[1]-a[0]) * offset / hBlkSize   (horizontal interp top row)
      x2 = a[2] + (a[3]-a[2]) * offset / hBlkSize   (horizontal interp bottom row)
      y  = x1 + (x2-x1) * row_in_vblk / vBlkSize    (vertical interp)
      interpolated_value[row, j*NPC+l] = y
    offset++; if offset == hBlkSize: offset=0, blk++

Final output computation (compute):
  For each pixel:
    max_log = max(log(B+eps), log(G+eps), log(R+eps))  [max over log channels]
    max_in  = max(B, G, R)  [integer max of raw channel values]
    iMin, iMax = bilinearly interpolated min/max of log-luminance (float32)
    computeVal = (max_log - iMin) / ((iMax - iMin) + eps)
    ratio = computeVal / (max_in + eps)
    For each channel c: out_c = clamp(channel_c * ratio * 255.0, 0, 255)
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
    """Unpack a bit-stream of pixel_bits-wide pixels from ptr_width_bits-wide words."""
    byte_data = words_u32.astype(np.uint32).tobytes()
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


def _float_to_raw_bits(f):
    return struct.unpack('<I', struct.pack('<f', np.float32(f)))[0]


def _raw_bits_to_float(u):
    return struct.unpack('<f', struct.pack('<I', int(u) & 0xFFFFFFFF))[0]


class LTMTile:
    """Replicates LTMTile with runtime blk_rows/blk_cols (the 4-arg constructor)."""

    def __init__(self, rows, cols, blk_rows, blk_cols, npc,
                 ct_blk_rows=BLOCK_HEIGHT, ct_blk_cols=BLOCK_WIDTH):
        self.rows = rows
        self.cols = cols
        self.blk_rows = blk_rows
        self.blk_cols = blk_cols
        self.npc = npc
        npc_shift = int(np.log2(npc))
        self.npc_shift = npc_shift

        self.cols_npc = (cols + npc - 1) >> npc_shift

        # Horizontal block geometry
        # blkColsLog2: start from log2(ct_blk_cols), increment until (1<<blkColsLog2) >= blk_cols
        ct_blk_cols_log2 = int(np.log2(ct_blk_cols))
        blk_cols_log2 = ct_blk_cols_log2
        cols_log2_max = int(np.log2(cols)) if cols > 0 else 0
        while blk_cols_log2 <= cols_log2_max:
            if (1 << blk_cols_log2) >= blk_cols:
                break
            blk_cols_log2 += 1
        self.blk_cols_log2 = blk_cols_log2
        self.blk_cols_npc = 1 << (blk_cols_log2 - npc_shift)

        th = ((self.cols_npc + (self.blk_cols_npc - 1)) >> blk_cols_log2)
        self.h_blk_count = th
        th_total = th << blk_cols_log2
        self.h_blk_size = [self.blk_cols_npc, self.blk_cols_npc]
        if self.cols_npc < th_total:
            t = self.cols_npc - ((self.cols_npc >> blk_cols_log2) << blk_cols_log2)
            self.h_blk_size[0] = t >> 1
            self.h_blk_count += 1

        # Vertical block geometry
        ct_blk_rows_log2 = int(np.log2(ct_blk_rows))
        blk_rows_log2 = ct_blk_rows_log2
        rows_log2_max = int(np.log2(rows)) if rows > 0 else 0
        while blk_rows_log2 <= rows_log2_max:
            if (1 << blk_rows_log2) >= blk_rows:
                break
            blk_rows_log2 += 1
        self.blk_rows_log2 = blk_rows_log2

        vc = (rows + (blk_rows - 1)) >> blk_rows_log2
        self.v_blk_count = vc
        tv = vc << blk_rows_log2
        self.v_blk_size = [blk_rows, blk_rows]
        if rows < tv:
            t = rows - ((rows >> blk_rows_log2) << blk_rows_log2)
            self.v_blk_size[0] = t >> 1
            self.v_blk_count += 1

        # MinMax array sizes (compile-time, matching C++ static constexpr)
        ct_blk_cols_npc = ct_blk_cols >> npc_shift
        cols_npc_ct = (WIDTH + npc - 1) >> npc_shift
        if (HEIGHT % ct_blk_rows) == 0:
            self.mm_v_size = (HEIGHT // ct_blk_rows) + 1
        else:
            self.mm_v_size = ((HEIGHT + ct_blk_rows - 1) // ct_blk_rows + 1) + 1
        if (cols_npc_ct % ct_blk_cols_npc) == 0:
            self.mm_h_size = (cols_npc_ct // ct_blk_cols_npc) + 1
        else:
            self.mm_h_size = ((cols_npc_ct + ct_blk_cols_npc - 1) // ct_blk_cols_npc + 1) + 1

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
        """j_npc is NPC-aligned column index (same as C++ getHBlkSize called with j).
        C++ getHBlkSize comment says 'addr is actual column number' but it's called
        with j (NPC-aligned index). For 512 cols with 128-wide blocks, mHBlkSize[0]=0
        so it always returns mHBlkSize[1]=32.
        We replicate C++ logic: compare j_npc against mHBlkSize[0] and
        (mColsNPCAligned - mHBlkSize[0])."""
        if j_npc < self.h_blk_size[0]:
            return self.h_blk_size[0]
        if j_npc >= (self.cols_npc - self.h_blk_size[0]):
            return self.h_blk_size[0]
        return self.h_blk_size[1]


def _ltm_process_minmax(max_log_npc, tile):
    """Compute per-block min/max of log-luminance, exactly matching processMinMax/getMinMax.

    max_log_npc[row, j, l] = max log luminance for pixel at (row, j*NPC+l).
    Returns omin_w, omax_w as 2D arrays of raw float bits, shape (mm_v_size, mm_h_size).

    Key: omin_l/omax_l persist across vBlkIndex iterations (NOT re-initialized per call).
    """
    HIGH = _float_to_raw_bits(float(np.log(float(0xFFFFFFFF))))
    LOW = _float_to_raw_bits(0.0)

    omin_w = np.zeros((tile.mm_v_size, tile.mm_h_size), dtype=np.uint32)
    omax_w = np.zeros((tile.mm_v_size, tile.mm_h_size), dtype=np.uint32)

    # omin_l and omax_l persist across all getMinMax calls
    omin_l = [HIGH] * tile.mm_h_size
    omax_l = [LOW] * tile.mm_h_size

    # Sequential row index into the image
    row_idx = 0
    H = tile.rows

    for vi in range(tile.v_blk_count + 1):
        v_blk_size = tile.get_minmax_v_blk_size(vi)
        if v_blk_size == 0:
            continue

        # bStart resets to True at start of each getMinMax call
        b_start = True

        for i in range(v_blk_size):
            row = row_idx + i
            if row >= H:
                break

            blk = 0
            offset = 0
            for j in range(tile.cols_npc):
                h_blk_size = tile.get_h_blk_size(j)

                # Compute min and max of max_log across NPC pixels at column group j
                pin_val_min = HIGH
                pin_val_max = LOW

                for l in range(NPC):
                    col = j * NPC + l
                    if col >= tile.cols:
                        break
                    val_bits = _float_to_raw_bits(max_log_npc[row, j, l])
                    val_f = _raw_bits_to_float(val_bits)

                    if val_f < _raw_bits_to_float(pin_val_min):
                        pin_val_min = val_bits
                    if val_f > _raw_bits_to_float(pin_val_max):
                        pin_val_max = val_bits

                cur_min = omin_l[blk]
                cur_max = omax_l[blk]

                if b_start:
                    cur_min = pin_val_min
                    cur_max = pin_val_min  # C++ sets max = pinValMin on bStart (not pinValMax)
                else:
                    if _raw_bits_to_float(pin_val_min) < _raw_bits_to_float(cur_min):
                        cur_min = pin_val_min
                    if _raw_bits_to_float(pin_val_max) > _raw_bits_to_float(cur_max):
                        cur_max = pin_val_max

                offset += 1
                b_start = False

                if offset == (h_blk_size >> 1):
                    blk += 1
                    b_start = (i == 0)  # True only on first row of this vBlk call

                if offset == h_blk_size:
                    offset = 0

                # Write to omin_l/omax_l at current blk (after possible blk increment)
                omin_l[blk] = cur_min
                omax_l[blk] = cur_max

                omin_w[vi, blk] = cur_min
                omax_w[vi, blk] = cur_max

        row_idx += v_blk_size

    return omin_w, omax_w


def _ltm_process_interpolate(arr, tile):
    """Bilinear interpolation of min/max array, matching processInterpolate/interpolate.

    arr: shape (mm_v_size, mm_h_size) of raw float bits (uint32).
    Returns interpolated float32 array of shape (H, W).
    """
    H, W = tile.rows, tile.cols
    out = np.zeros((H, W), dtype=np.float32)

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

                # 2x2 neighborhood: [vi+m][blk+n] for m,n in {0,1}
                a = [0.0] * 4
                for m in range(2):
                    for n in range(2):
                        idx_v = vi + m
                        idx_h = blk + n
                        if idx_v < tile.mm_v_size and idx_h < tile.mm_h_size:
                            a[m * 2 + n] = _raw_bits_to_float(arr[idx_v, idx_h])
                        else:
                            a[m * 2 + n] = 0.0

                for l in range(NPC):
                    col = j * NPC + l
                    if col >= W:
                        break
                    # Horizontal interpolation (same for all l in this NPC group)
                    x1 = a[0] + (a[1] - a[0]) * offset / h_blk_size
                    x2 = a[2] + (a[3] - a[2]) * offset / h_blk_size
                    # Vertical interpolation
                    y = x1 + (x2 - x1) * i / v_blk_size
                    out[row_idx, col] = np.float32(y)

                offset += 1
                if offset == h_blk_size:
                    offset = 0
                    blk += 1

            row_idx += 1

    return out


def _ltm_single_pass(pixels_bgr, tile, omin_r, omax_r):
    """Run one pass of LTM. Returns (output_pixels uint8 HxWx3, omin_w, omax_w).

    pixels_bgr: HxWx3 uint8 array.
    omin_r, omax_r: (mm_v_size, mm_h_size) uint32 arrays of raw float bits (from previous pass).
    """
    H, W = tile.rows, tile.cols

    # Compute per-channel log values (as float32, matching C++ float arithmetic)
    b = pixels_bgr[:, :, 0].astype(np.float32)
    g = pixels_bgr[:, :, 1].astype(np.float32)
    r = pixels_bgr[:, :, 2].astype(np.float32)

    log_b = np.log(b + EPS).astype(np.float32)
    log_g = np.log(g + EPS).astype(np.float32)
    log_r = np.log(r + EPS).astype(np.float32)

    # max_log: per-pixel maximum of log channels (float32)
    max_log = np.maximum(np.maximum(log_b, log_g), log_r).astype(np.float32)

    # max_in: integer max of raw channels
    max_in = np.maximum(np.maximum(
        pixels_bgr[:, :, 0].astype(np.float32),
        pixels_bgr[:, :, 1].astype(np.float32)),
        pixels_bgr[:, :, 2].astype(np.float32))

    # Reshape max_log into NPC groups: (H, cols_npc, NPC)
    # Pad W to be a multiple of NPC
    W_padded = tile.cols_npc * NPC
    max_log_padded = np.zeros((H, W_padded), dtype=np.float32)
    max_log_padded[:, :W] = max_log
    max_log_npc = max_log_padded.reshape(H, tile.cols_npc, NPC)

    # Compute omin_w, omax_w for this pass
    omin_w, omax_w = _ltm_process_minmax(max_log_npc, tile)

    # Bilinearly interpolate omin_r and omax_r from the previous pass
    interp_min = _ltm_process_interpolate(omin_r, tile)
    interp_max = _ltm_process_interpolate(omax_r, tile)

    # Compute output pixels
    out_pixels = np.zeros((H, W, 3), dtype=np.uint8)

    for row in range(H):
        for col in range(W):
            ml = float(max_log[row, col])
            mi = float(max_in[row, col])
            i_min = float(interp_min[row, col])
            i_max = float(interp_max[row, col])

            # compute: (max_log - iMin) / ((iMax - iMin) + eps)
            compute_val = (ml - i_min) / ((i_max - i_min) + float(EPS))
            # ratio: computeVal / (max_in + eps)
            ratio = compute_val / (mi + float(EPS))

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

    # Extract BGR channels
    b = (pixel_data & 0xFF).astype(np.uint8)
    g = ((pixel_data >> 8) & 0xFF).astype(np.uint8)
    r = ((pixel_data >> 16) & 0xFF).astype(np.uint8)

    pixels_bgr = np.stack([b.reshape(HEIGHT, WIDTH),
                           g.reshape(HEIGHT, WIDTH),
                           r.reshape(HEIGHT, WIDTH)], axis=2)

    tile = LTMTile(HEIGHT, WIDTH, BLK_HEIGHT_RT, BLK_WIDTH_RT, NPC,
                   ct_blk_rows=BLOCK_HEIGHT, ct_blk_cols=BLOCK_WIDTH)

    # Initialize omin/omax to all-zeros (simulating static zero-initialization in HLS)
    omin_init = np.zeros((tile.mm_v_size, tile.mm_h_size), dtype=np.uint32)
    omax_init = np.zeros((tile.mm_v_size, tile.mm_h_size), dtype=np.uint32)

    # Pass 1: warmup — reads zeros, writes omin_1/omax_1; output pixels discarded
    _, omin_1, omax_1 = _ltm_single_pass(pixels_bgr, tile, omin_init, omax_init)

    # Pass 2: actual — reads omin_1/omax_1, produces final output
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
