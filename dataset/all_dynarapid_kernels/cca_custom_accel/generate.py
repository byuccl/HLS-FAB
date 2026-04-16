import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
TOTAL_ELEMENTS = HEIGHT * WIDTH

function_signature = "void cca_custom_accel(uint8_t* in_ptr1, uint8_t* in_ptr2, uint8_t* tmp_out_ptr1, uint8_t* tmp_out_ptr2, uint8_t* out_ptr, int32_t* obj_pix, int32_t* def_pix)"

description = f"""Custom Connected Component Analysis (CCA) for defect detection on a binary image.

The function takes two copies of the same binary input image ({HEIGHT}x{WIDTH}, {TOTAL_ELEMENTS} uint8 pixels).
Pixel value 0 indicates an object/foreground pixel. Non-zero indicates background.

The algorithm performs two passes:
1. Forward pass (top-to-bottom, left-to-right) on in_ptr1, and a reverse pass (bottom-to-top,
   right-to-left) on in_ptr2. Each pass processes each pixel using a neighborhood connectivity
   check with a sliding label array. For each foreground pixel (non-zero input), if any of the
   4 neighboring labels (left-neighbor from previous row, current from previous row,
   right-neighbor from previous row, or left-neighbor in current row) is 1, the pixel is
   labeled as connected (output 0). Otherwise, the pixel is labeled as a defect (output 255).
   Background pixels (input == 0) always output 0 and are counted as object pixels.

2. The forward and reverse pass results are combined using bitwise AND. Pixels that are 255
   in both passes are defect pixels. The output image out_ptr contains the AND result.

Outputs:
- tmp_out_ptr1: temporary buffer with forward pass results ({TOTAL_ELEMENTS} uint8)
- tmp_out_ptr2: temporary buffer with reverse pass results ({TOTAL_ELEMENTS} uint8)
- out_ptr: final defect image ({TOTAL_ELEMENTS} uint8, values 0 or 255)
- obj_pix: count of object pixels (where input == 0) from the forward pass (1 int32)
- def_pix: count of defect pixels (non-zero in out_ptr) (1 int32)
"""

inputs_info = [
    {
        "name": "in_ptr1",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS,),
        "total_elements": TOTAL_ELEMENTS,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "in_ptr2",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS,),
        "total_elements": TOTAL_ELEMENTS,
        "dtype": "int32",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "tmp_out_ptr1",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS,),
        "total_elements": TOTAL_ELEMENTS,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "tmp_out_ptr2",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS,),
        "total_elements": TOTAL_ELEMENTS,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "out_ptr",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS,),
        "total_elements": TOTAL_ELEMENTS,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "obj_pix",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "out",
    },
    {
        "name": "def_pix",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int32",
        "iotype": "out",
    },
]


def _process_row(in_row, tmp_out_row, lab_arr, width):
    """Process one row for CCA. Returns obj_pix count for this row."""
    obj_pix = 0
    a = True
    b = lab_arr[0]
    c = lab_arr[1]
    d = True

    for j in range(width):
        if in_row[j] != 0:
            if a or b or c or d:
                lab = True
                out = 0
            else:
                lab = False
                out = 255
        else:
            lab = False
            out = 0
            obj_pix += 1

        lab_arr[j] = lab
        tmp_out_row[j] = out
        a = b
        b = c
        c = lab_arr[j + 2]
        d = lab

    return obj_pix


def behavioral(in_ptr1, in_ptr2):
    height = HEIGHT
    width = WIDTH

    in1 = in_ptr1.reshape(height, width).copy()
    in2 = in_ptr2.reshape(height, width).copy()

    # Forward CCA pass
    tmp_out1 = np.zeros((height, width), dtype=np.uint8)
    obj_pix_total = 0
    lab_arr = [True] * (width + 2)

    for i in range(height):
        row_in = in1[i, :]
        row_out = np.zeros(width, dtype=np.uint8)
        obj_count = _process_row(row_in, row_out, lab_arr, width)
        tmp_out1[i, :] = row_out
        obj_pix_total += obj_count

    # Reverse CCA pass
    tmp_out2_flat = np.zeros(height * width, dtype=np.uint8)
    in2_flat = in2.flatten()
    lab_arr_rev = [True] * (width + 2)

    # The reverse pass processes in2 from bottom to top using a double-buffering scheme.
    rd_linebuff1 = np.zeros(width, dtype=np.uint8)
    rd_linebuff2 = np.zeros(width, dtype=np.uint8)
    wrt_linebuff1 = np.zeros(width, dtype=np.uint8)
    wrt_linebuff2 = np.zeros(width, dtype=np.uint8)

    rd_offset = height * width
    wrt_offset = height * width

    # Read last row into rd_linebuff1
    rd_offset -= width
    rd_linebuff1[:] = in2_flat[rd_offset:rd_offset + width]

    # Read second-to-last row into rd_linebuff2
    rd_offset -= width
    rd_linebuff2[:] = in2_flat[rd_offset:rd_offset + width]

    # Process last row
    _process_row(rd_linebuff1, wrt_linebuff1, lab_arr_rev, width)

    flag = False

    for i in range(height - 2):
        rd_offset -= width
        wrt_offset -= width

        if not flag:
            rd_linebuff1[:] = in2_flat[rd_offset:rd_offset + width]
            _process_row(rd_linebuff2, wrt_linebuff2, lab_arr_rev, width)
            tmp_out2_flat[wrt_offset:wrt_offset + width] = wrt_linebuff1[:]
            flag = True
        else:
            rd_linebuff2[:] = in2_flat[rd_offset:rd_offset + width]
            _process_row(rd_linebuff1, wrt_linebuff1, lab_arr_rev, width)
            tmp_out2_flat[wrt_offset:wrt_offset + width] = wrt_linebuff2[:]
            flag = False

    wrt_offset -= width

    if not flag:
        _process_row(rd_linebuff2, wrt_linebuff2, lab_arr_rev, width)
        tmp_out2_flat[wrt_offset:wrt_offset + width] = wrt_linebuff1[:]
        flag = True
    else:
        _process_row(rd_linebuff1, wrt_linebuff1, lab_arr_rev, width)
        tmp_out2_flat[wrt_offset:wrt_offset + width] = wrt_linebuff2[:]
        flag = False

    wrt_offset -= width

    if not flag:
        tmp_out2_flat[wrt_offset:wrt_offset + width] = wrt_linebuff1[:]
    else:
        tmp_out2_flat[wrt_offset:wrt_offset + width] = wrt_linebuff2[:]

    # Pass 2: AND the two temp outputs
    out = (tmp_out1.flatten() & tmp_out2_flat).astype(np.uint8)
    def_pix_total = int(np.count_nonzero(out))

    return (
        tmp_out1.astype(np.uint8).flatten(),
        tmp_out2_flat,
        out,
        np.array([obj_pix_total], dtype=np.int32),
        np.array([def_pix_total], dtype=np.int32),
    )


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
