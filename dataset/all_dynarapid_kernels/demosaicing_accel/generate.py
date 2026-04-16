import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = 128
WIDTH = 128
TOTAL_PIXELS = HEIGHT * WIDTH
# Input: 1 byte per pixel (XF_8UC1)
# Output: 3 bytes per pixel (XF_8UC3), packed as B, G, R
TOTAL_OUT_BYTES = TOTAL_PIXELS * 3

# Hardcoded Bayer pattern: XF_BAYER_RG
# RG pattern:
#   Row 0: R G R G ...
#   Row 1: G B G B ...
BAYER_RG = 3

function_signature = "void demosaicing_accel(int32_t* img_in, int32_t* img_out)"

description = f"""Bayer demosaicing using the Malvar-He-Cutler algorithm on a {HEIGHT}x{WIDTH}
single-channel 8-bit Bayer (RG pattern) image, producing a 3-channel 8-bit color image.

Input: {TOTAL_PIXELS} uint8 values representing the Bayer mosaic image (one byte per pixel,
row-major).

Output: {TOTAL_OUT_BYTES} uint8 values representing the demosaiced color image. Each pixel
produces 3 bytes in B, G, R order, tightly packed (pixel 0 at bytes 0-2, pixel 1 at bytes 3-5,
etc.).

The RG Bayer pattern is:
  Row 0: R  G  R  G  ...
  Row 1: G  B  G  B  ...
  Row 2: R  G  R  G  ...

For each pixel, the two missing color channels are interpolated from a 5x5 neighborhood using
four integer-arithmetic kernel functions:

1. g_kernel (Green at a Red or Blue location):
   res = -(top2 + left2 + right2 + bottom2) + 2*(top1 + left1 + right1 + bottom1) + 4*center
   res = res / 8 (integer division); if res < 0, res = 0

2. rb_kernel (Red at Blue or Blue at Red location):
   t1 = (top2 + left2 + right2 + bottom2) * 3 / 2 (integer division)
   t2 = (diag_tl + diag_tr + diag_bl + diag_br) * 2
   t3 = center * 6
   res = (-t1 + t2 + t3) / 8; if res < 0, res = 0

3. rgr_bgb_kernel (Red at Green in Red row, or Blue at Green in Blue row):
   t1 = (top2 + bottom2) >> 1 (arithmetic right shift)
   t2 = diag_tl + diag_tr + left2 + right2 + diag_bl + diag_br
   t3 = (left1 + right1) * 4
   t4 = center * 5
   res = (t1 - t2 + t3 + t4) / 8; if res < 0, res = 0

4. rgb_bgr_kernel (Red at Green in Blue row, or Blue at Green in Red row):
   t1 = (left2 + right2) / 2 (integer division)
   t2 = top2 + diag_tl + diag_tr + diag_bl + diag_br + bottom2
   t3 = (top1 + bottom1) * 4
   t4 = center * 5
   res = (t1 - t2 + t3 + t4) / 8; if res < 0, res = 0

Here, top2/bottom2/left2/right2 are 2 pixels away from center in the 5x5 neighborhood,
top1/bottom1/left1/right1 are 1 pixel away, and diag_XX are diagonal neighbors.

Out-of-bounds pixels are treated as zero (constant border padding with 2-pixel border).
Output channels are clamped to [0, 255].
"""

inputs_info = [
    {
        "name": "img_in",
        "iotype": "in",
        "array_dimension": 1,
        "dtype": "int32",
        "shape": (TOTAL_PIXELS,),
        "total_elements": TOTAL_PIXELS,
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "iotype": "out",
        "array_dimension": 1,
        "dtype": "int32",
        "shape": (TOTAL_OUT_BYTES,),
        "total_elements": TOTAL_OUT_BYTES,
        "min": 0,
        "max": 255,
    },
]


def behavioral(img_in):
    # Implements Vitis xf::cv::demosaicing with XF_BAYER_RG (=3), NPC=1
    # Uses zero-padding (XF_BORDER_CONSTANT) with 2-pixel border.
    # Output pixel format: B in byte[0], G in byte[1], R in byte[2] (matching HLS word packing).
    #
    # XF_BAYER_RG pattern (row/col offset +1 applied to candidateRow/candidateCol in Core_Process):
    #   Row 0: R G R G ...   (candidateRow=i+1 odd -> R row)
    #   Row 1: G B G B ...   (candidateRow=i+1 even -> B row)
    #
    # For each pixel (i,j), candidateRow = i+1, candidateCol = j+1:
    #   (i+1) odd  & (j+1) odd  = i even, j even -> R pixel: b=rb_kernel, g=g_kernel, r=center
    #   (i+1) odd  & (j+1) even = i even, j odd  -> G in R row: b=rgb_bgr_kernel, g=center, r=rgr_bgb_kernel
    #   (i+1) even & (j+1) odd  = i odd,  j even -> G in B row: b=rgr_bgb_kernel, g=center, r=rgb_bgr_kernel
    #   (i+1) even & (j+1) even = i odd,  j odd  -> B pixel: r=rb_kernel, g=g_kernel, b=center
    #
    # Kernel uses imgblock[5][buf_size] with 2-pixel zero padding (rows 0,1 and cols 0,1 of 5x5 window).
    # imgblock[2][2+loop] is the center pixel (loop=0 for NPC=1).
    # Indexing for center at (i,j):
    #   imgblock[row][col+2] = padded_image[i + row - 2][j + col - 2]
    # where padded_image has 2-pixel zero border.

    H, W = HEIGHT, WIDTH
    img = img_in.reshape(H, W).astype(np.int32)

    # Zero-pad with 2 pixels on each side
    pad = np.pad(img, 2, mode='constant', constant_values=0)

    # Extract 5x5 neighborhood elements needed by kernels
    # For pixel at (i,j) in original image, center is pad[i+2, j+2]
    # imgblock[r][c+2] = pad[i + r, j + c]  (r in 0..4, c in 0..4)
    # Key positions (r, c offsets from center at (2,2) in 5x5 block):
    #   top2    = imgblock[0][2] = pad[i,   j+2] -> pad[i+0, j+2]
    #   top1    = imgblock[1][2] = pad[i+1, j+2]
    #   center  = imgblock[2][2] = pad[i+2, j+2]
    #   bottom1 = imgblock[3][2] = pad[i+3, j+2]
    #   bottom2 = imgblock[4][2] = pad[i+4, j+2]
    #   left2   = imgblock[2][0] = pad[i+2, j+0]
    #   left1   = imgblock[2][1] = pad[i+2, j+1]
    #   right1  = imgblock[2][3] = pad[i+2, j+3]
    #   right2  = imgblock[2][4] = pad[i+2, j+4]
    #   diag_tl = imgblock[1][1] = pad[i+1, j+1]
    #   diag_tr = imgblock[1][3] = pad[i+1, j+3]
    #   diag_bl = imgblock[3][1] = pad[i+3, j+1]
    #   diag_br = imgblock[3][3] = pad[i+3, j+3]
    #
    # All indexing uses slices over the full (H,W) grid.

    # Extract all relevant neighbors as (H,W) arrays using slice notation on pad
    top2    = pad[0:H,   2:W+2]
    top1    = pad[1:H+1, 2:W+2]
    center  = pad[2:H+2, 2:W+2]
    bottom1 = pad[3:H+3, 2:W+2]
    bottom2 = pad[4:H+4, 2:W+2]
    left2   = pad[2:H+2, 0:W]
    left1   = pad[2:H+2, 1:W+1]
    right1  = pad[2:H+2, 3:W+3]
    right2  = pad[2:H+2, 4:W+4]
    diag_tl = pad[1:H+1, 1:W+1]
    diag_tr = pad[1:H+1, 3:W+3]
    diag_bl = pad[3:H+3, 1:W+1]
    diag_br = pad[3:H+3, 3:W+3]

    # g_kernel: -(top2+left2+right2+bottom2) + 2*(top1+left1+right1+bottom1) + 4*center, /8, clamp 0
    g_k = (-(top2 + left2 + right2 + bottom2)
           + 2 * (top1 + left1 + right1 + bottom1)
           + 4 * center)
    g_k = g_k // 8
    g_k = np.where(g_k < 0, 0, g_k)

    # rb_kernel: (-t1 + t2 + t3) / 8 where
    #   t1 = (top2+left2+right2+bottom2)*3/2 (integer division)
    #   t2 = (diag_tl+diag_tr+diag_bl+diag_br)*2
    #   t3 = center*6
    t1 = (top2 + left2 + right2 + bottom2) * 3 // 2
    t2 = (diag_tl + diag_tr + diag_bl + diag_br) * 2
    t3 = center * 6
    rb_k = (-t1 + t2 + t3) // 8
    rb_k = np.where(rb_k < 0, 0, rb_k)

    # rgr_bgb_kernel: ((top2+bottom2)>>1 - t2 + t3 + t4) / 8 where
    #   t2 = diag_tl+diag_tr+left2+right2+diag_bl+diag_br
    #   t3 = (left1+right1)*4
    #   t4 = center*5
    t1_rgr = (top2 + bottom2) >> 1
    t2_rgr = diag_tl + diag_tr + left2 + right2 + diag_bl + diag_br
    t3_rgr = (left1 + right1) * 4
    t4_rgr = center * 5
    rgr_bgb_k = (t1_rgr - t2_rgr + t3_rgr + t4_rgr) // 8
    rgr_bgb_k = np.where(rgr_bgb_k < 0, 0, rgr_bgb_k)

    # rgb_bgr_kernel: (t1 - t2 + t3 + t4) / 8 where
    #   t1 = (left2+right2)/2
    #   t2 = top2+diag_tl+diag_tr+diag_bl+diag_br+bottom2
    #   t3 = (top1+bottom1)*4
    #   t4 = center*5
    t1_rgb = (left2 + right2) // 2
    t2_rgb = top2 + diag_tl + diag_tr + diag_bl + diag_br + bottom2
    t3_rgb = (top1 + bottom1) * 4
    t4_rgb = center * 5
    rgb_bgr_k = (t1_rgb - t2_rgb + t3_rgb + t4_rgb) // 8
    rgb_bgr_k = np.where(rgb_bgr_k < 0, 0, rgb_bgr_k)

    # Build row/col index parity masks based on candidateRow=i+1, candidateCol=j+1
    # (i+1) is odd iff i is even; (j+1) is odd iff j is even
    rows = np.arange(H).reshape(H, 1)
    cols = np.arange(W).reshape(1, W)
    i_even = (rows % 2 == 0)  # candidateRow odd -> R row
    j_even = (cols % 2 == 0)  # candidateCol odd

    # XF_BAYER_RG with candidateRow=i+1, candidateCol=j+1:
    # cRow odd (i even) & cCol odd (j even): R pixel   -> r=center, g=g_k, b=rb_k
    # cRow odd (i even) & cCol even (j odd): G in R row-> r=rgr_bgb_k, g=center, b=rgb_bgr_k
    # cRow even (i odd) & cCol odd (j even): G in B row-> r=rgb_bgr_k, g=center, b=rgr_bgb_k
    # cRow even (i odd) & cCol even (j odd): B pixel   -> r=rb_k, g=g_k, b=center

    # Compute R, G, B planes
    R_out = np.where(i_even & j_even, center,
            np.where(i_even & ~j_even, rgr_bgb_k,
            np.where(~i_even & j_even, rgb_bgr_k,
                     rb_k)))

    G_out = np.where(i_even & j_even, g_k,
            np.where(i_even & ~j_even, center,
            np.where(~i_even & j_even, center,
                     g_k)))

    B_out = np.where(i_even & j_even, rb_k,
            np.where(i_even & ~j_even, rgb_bgr_k,
            np.where(~i_even & j_even, rgr_bgb_k,
                     center)))

    # Saturate to [0, 255] — xf_satcast only clamps at max (not at 0 for negative),
    # but kernels already return 0 for negative; center values are always in [0,255].
    R_out = np.clip(R_out, 0, 255).astype(np.uint8)
    G_out = np.clip(G_out, 0, 255).astype(np.uint8)
    B_out = np.clip(B_out, 0, 255).astype(np.uint8)

    # Output packed as B, G, R per pixel (HLS word: bits[7:0]=B, [15:8]=G, [23:16]=R)
    out = np.stack([B_out, G_out, R_out], axis=-1)
    return out.flatten()


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
