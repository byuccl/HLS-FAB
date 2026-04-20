import numpy as np
from hlsfab import PromptConstructor, generate_examples

function_signature = "void laplace_filter(ap_uint<32> *in_pixels, ap_uint<32> *out_pixel)"

short_description = "This HLS kernel implements a function to perform a Laplacian filter on a cluster of nine pixels of an image."

description = """
This HLS kernel applies a 3×3 Laplacian filter to a neighborhood of nine pixels from an image.

Interface and pixel format:
- The function takes two parameters: an input pointer to nine 32‑bit words (the 3×3 pixel block in row-major order:
  (1,1), (1,2), (1,3), (2,1), ..., (3,3)), and an output pointer to a single 32‑bit word.
- Each 32‑bit word encodes one RGB pixel in 0x00RRGGBB format: bits [7:0] blue, [15:8] green, [23:16] red,
  and bits [31:24] are unused.

Grayscale conversion:
- Each input pixel is first converted to an 8‑bit grayscale intensity using an integer luminance approximation:
  gray = (77 * R + 150 * G + 29 * B) >> 8. [web:59][web:51]

Laplacian filter:
- The kernel then applies a discrete 3×3 Laplacian using only the center pixel and its four direct neighbors
  (north, west, east, south) with the stencil:

    0  -1   0
   -1   4  -1
    0  -1   0

- In flat array form, the coefficients are {0, -1, 0, -1, 4, -1, 0, -1, 0}, but only indices 1, 3, 4, 5, and 7
  contribute to the sum.

Post-processing and output:
- The Laplacian sum is reduced to 8 bits by masking with 0xFF (modulo 256 wrap, not saturation).
- This 8‑bit value is then replicated into the R, G, and B channels and packed back into a 32‑bit 0x00RRGGBB word,
  which is written to the output pointer.
"""


inputs_info = [
    {
        "name": "in_pixels",
        "array_dimension": 1,
        "shape": (9,),
        "total_elements": 9,
        "dtype": "uint32",
        "iotype": "in",
        "min": 0,
        "max": 16777215,
    },
    {
        "name": "out_pixel",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint32",
        "iotype": "out",
    }
]

def behavioral(in_pixels):
    """
    in_pixels: flat iterable of 9 ints, each 0xRRGGBB
               index layout:
                 0 1 2
                 3 4 5
                 6 7 8
    """
    p = in_pixels  # alias

    gray_1_1 = ((p[0] >> 16) & 0xFF) * 77 + ((p[0] >> 8) & 0xFF) * 150 + (p[0] & 0xFF) * 29
    gray_1_1 >>= 8

    gray_1_2 = ((p[1] >> 16) & 0xFF) * 77 + ((p[1] >> 8) & 0xFF) * 150 + (p[1] & 0xFF) * 29
    gray_1_2 >>= 8

    gray_1_3 = ((p[2] >> 16) & 0xFF) * 77 + ((p[2] >> 8) & 0xFF) * 150 + (p[2] & 0xFF) * 29
    gray_1_3 >>= 8

    gray_2_1 = ((p[3] >> 16) & 0xFF) * 77 + ((p[3] >> 8) & 0xFF) * 150 + (p[3] & 0xFF) * 29
    gray_2_1 >>= 8

    gray_2_2 = ((p[4] >> 16) & 0xFF) * 77 + ((p[4] >> 8) & 0xFF) * 150 + (p[4] & 0xFF) * 29
    gray_2_2 >>= 8

    gray_2_3 = ((p[5] >> 16) & 0xFF) * 77 + ((p[5] >> 8) & 0xFF) * 150 + (p[5] & 0xFF) * 29
    gray_2_3 >>= 8

    gray_3_1 = ((p[6] >> 16) & 0xFF) * 77 + ((p[6] >> 8) & 0xFF) * 150 + (p[6] & 0xFF) * 29
    gray_3_1 >>= 8

    gray_3_2 = ((p[7] >> 16) & 0xFF) * 77 + ((p[7] >> 8) & 0xFF) * 150 + (p[7] & 0xFF) * 29
    gray_3_2 >>= 8

    gray_3_3 = ((p[8] >> 16) & 0xFF) * 77 + ((p[8] >> 8) & 0xFF) * 150 + (p[8] & 0xFF) * 29
    gray_3_3 >>= 8

    laplacian = -gray_1_2 + -gray_2_1 + 4 * gray_2_2 + -gray_2_3 + -gray_3_2
    laplacian &= 0xFF
    return_array = np.array([(laplacian << 16) | (laplacian << 8) | laplacian])
    return return_array

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")