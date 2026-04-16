import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 676
WIDTH = 1024

SCALE_FACTOR = 256
MAX_REPRESENTED_VALUE = 65536
LOG2_SCALE_FACTOR = 8
LOG2_QUANTIZATION_INTERVAL = 8  # log2(MAX_REPRESENTED_VALUE / SCALE_FACTOR) = log2(256)
BITDEPTH_IN = 16
BITDEPTH_OUT = 8
PLANES = 3

function_signature = "void quantizationdithering_accel(int32_t* img_inp, int32_t* img_out)"

description = f"""Performs quantization dithering on a 3-channel RGB image, converting from 16-bit to 8-bit per channel.
The image has {HEIGHT} rows and {WIDTH} columns with {PLANES} interleaved channels.
The algorithm applies error diffusion dithering. For each pixel and channel:
1. A quantizer input is computed by adding the pixel value, an offset from the previous row's error diffusion, and 7/16 of the previous pixel's quantization error (computed via multiply-by-7 then arithmetic-right-shift-by-4).
2. The quantizer input is divided by the quantization interval ({MAX_REPRESENTED_VALUE // SCALE_FACTOR}) with rounding (add bit below truncation point) to produce an output level.
3. The quantization error is diffused: the lower bits of the quantizer input form a local error, which combines with prior errors using weights 1/16, 5/16, 3/16 to update an offset buffer for the next row.
4. Output is clamped to [0, {SCALE_FACTOR - 1}]. Negative values clamp to 0, values >= {SCALE_FACTOR} clamp to {SCALE_FACTOR - 1}.
Input is a flat array of {HEIGHT * WIDTH * PLANES} uint16 values in row-major, channel-interleaved order.
Output is a flat array of {HEIGHT * WIDTH * PLANES} uint8 values in the same order."""

inputs_info = [
    {
        "name": "img_inp",
        "iotype": "in",
        "array_dimension": 3,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH, PLANES),
        "total_elements": HEIGHT * WIDTH * PLANES,
        "min": 0,
        "max": 65535,
    },
    {
        "name": "img_out",
        "iotype": "out",
        "array_dimension": 3,
        "dtype": "int32",
        "shape": (HEIGHT, WIDTH, PLANES),
        "total_elements": HEIGHT * WIDTH * PLANES,
    },
]


def _to_signed(val, bits):
    """Truncate value to 'bits' width and interpret as signed."""
    val = int(val)
    mask = (1 << bits) - 1
    val = val & mask
    if val & (1 << (bits - 1)):
        val -= (1 << bits)
    return val


def behavioral(img_inp):
    """Quantization dithering: 16-bit to 8-bit with error diffusion."""
    inp = img_inp.astype(np.int64).reshape(HEIGHT, WIDTH, PLANES)
    out = np.zeros((HEIGHT, WIDTH, PLANES), dtype=np.uint8)

    offset_buffer = np.zeros((PLANES, WIDTH), dtype=np.int64)

    for row in range(HEIGHT):
        q_err_1st = np.zeros(PLANES, dtype=np.int64)
        q_err_2nd = np.zeros(PLANES, dtype=np.int64)
        offset_prev_NPC = np.zeros(PLANES, dtype=np.int64)

        # Process WIDTH real columns + 1 extra column (for final offset_buffer write)
        for col in range(WIDTH + 1):
            is_extra = (col == WIDTH)

            for ch in range(PLANES):
                # For extra column, use stale pixel (last column's value)
                if not is_extra:
                    in_pixel = int(inp[row, col, ch])
                else:
                    in_pixel = int(inp[row, WIDTH - 1, ch])

                # Read offset from buffer (extra col is OOB, treat as 0)
                if col < WIDTH:
                    ob_val = offset_buffer[ch, col]
                else:
                    ob_val = 0

                # 7/16 of previous pixel's error
                q_2nd_err_scale7 = _to_signed(q_err_2nd[ch] * 7, BITDEPTH_IN + 2)
                q_2nd_err_scale7by16 = _to_signed(q_2nd_err_scale7 >> 4, BITDEPTH_IN - 2)

                # Quantizer input
                quatizer_in = _to_signed(int(ob_val) + in_pixel + q_2nd_err_scale7by16, BITDEPTH_IN + 2)

                # Rounding division by quantization interval
                upper = _to_signed(quatizer_in >> LOG2_QUANTIZATION_INTERVAL, LOG2_SCALE_FACTOR + 2)
                rounding_bit = (quatizer_in >> (LOG2_QUANTIZATION_INTERVAL - 1)) & 1
                round_out = _to_signed(upper + rounding_bit, LOG2_SCALE_FACTOR + 2)

                # Local quantization error (lower bits, sign-extended)
                lower_mask = (1 << LOG2_QUANTIZATION_INTERVAL) - 1
                q_err_3rd_local = quatizer_in & lower_mask
                if q_err_3rd_local & (1 << (LOG2_QUANTIZATION_INTERVAL - 1)):
                    q_err_3rd_local -= (1 << LOG2_QUANTIZATION_INTERVAL)

                # Compute offset for next row
                sum_tmp = _to_signed(q_err_1st[ch] + q_err_2nd[ch] * 5 + q_err_3rd_local * 3, BITDEPTH_IN + 4)
                offset_NPC = _to_signed(sum_tmp >> 4, BITDEPTH_IN)

                # Write offset_buffer for previous column
                if col != 0:
                    write_col = col - 1
                    if write_col < WIDTH:
                        offset_buffer[ch, write_col] = offset_prev_NPC[ch]

                # Error propagation
                if is_extra:
                    q_err_1st[ch] = 0
                    q_err_2nd[ch] = 0
                else:
                    q_err_1st[ch] = q_err_2nd[ch]
                    q_err_2nd[ch] = q_err_3rd_local

                # Output clamping
                if is_extra or (round_out < 0):
                    out_val = 0
                elif round_out >= SCALE_FACTOR:
                    out_val = SCALE_FACTOR - 1
                else:
                    out_val = round_out & 0xFF

                if not is_extra:
                    out[row, col, ch] = out_val

                offset_prev_NPC[ch] = offset_NPC

    return out.astype(np.uint8).flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
