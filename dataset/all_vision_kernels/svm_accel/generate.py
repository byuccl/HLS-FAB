import numpy as np
from hlsfab import PromptConstructor, generate_examples

IN_ARRAY_SIZE_1 = 200
IN_ARRAY_SIZE_2 = 200
TOTAL_ELEMENTS_1 = IN_ARRAY_SIZE_1 * IN_ARRAY_SIZE_1  # 40000
TOTAL_ELEMENTS_2 = IN_ARRAY_SIZE_2 * IN_ARRAY_SIZE_2  # 40000
NO_OF_KERNEL_ELEMENTS = 200
INDEX_ARR_1 = 0
INDEX_ARR_2 = 0
IN_FRAC_BITS_1 = 15
IN_FRAC_BITS_2 = 15

function_signature = "void svm_accel(ap_uint<16>* img_in1, ap_uint<16>* img_in2, unsigned short* params, unsigned char* fractional_out, ap_int<64>* result_out)"

description = f"""Computes the dot product of two signed 16-bit fixed-point input arrays.

The function takes two input arrays of {TOTAL_ELEMENTS_1} signed 16-bit elements each, along with a
parameter array of 5 unsigned 16-bit values: [index1, index2, frac1, frac2, n].
- index1, index2: starting indices into the input arrays
- frac1, frac2: number of fractional bits in each input array
- n: number of elements to process in the dot product

The kernel reads n elements from each input array (starting at the respective index),
interprets them as signed 16-bit fixed-point values, multiplies corresponding pairs,
and accumulates the products into a 64-bit signed result.

It also outputs fractional_out = frac1 + frac2, which is the number of fractional bits
in the result.

Inputs:
- img_in1: array of {TOTAL_ELEMENTS_1} elements, each ap_uint<16> (interpreted as signed int16)
- img_in2: array of {TOTAL_ELEMENTS_2} elements, each ap_uint<16> (interpreted as signed int16)
- params: array of 5 unsigned short values [index1, index2, frac1, frac2, n]

Outputs:
- fractional_out: 1 unsigned char value equal to frac1 + frac2
- result_out: 1 signed 64-bit integer holding the dot product result
"""

inputs_info = [
    {
        "name": "img_in1",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS_1,),
        "total_elements": TOTAL_ELEMENTS_1,
        "dtype": "int16",
        "iotype": "in",
        "min": -32768,
        "max": 32767,
    },
    {
        "name": "img_in2",
        "array_dimension": 1,
        "shape": (TOTAL_ELEMENTS_2,),
        "total_elements": TOTAL_ELEMENTS_2,
        "dtype": "int16",
        "iotype": "in",
        "min": -32768,
        "max": 32767,
    },
    {
        "name": "params",
        "array_dimension": 1,
        "shape": (5,),
        "total_elements": 5,
        "dtype": "uint16",
        "iotype": "in",
        "min": 0,
        "max": 200,
    },
    {
        "name": "fractional_out",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "uint8",
        "iotype": "out",
    },
    {
        "name": "result_out",
        "array_dimension": 1,
        "shape": (1,),
        "total_elements": 1,
        "dtype": "int64",
        "iotype": "out",
    },
]


def behavioral(img_in1, img_in2, params):
    # Interpret img_in1 and img_in2 as signed 16-bit
    arr1 = img_in1.astype(np.int16)
    arr2 = img_in2.astype(np.int16)

    index1 = int(params[0])
    index2 = int(params[1])
    frac1 = int(params[2])
    frac2 = int(params[3])
    n = int(params[4])

    # Clamp n to not exceed array bounds
    n1 = min(n, len(arr1) - index1)
    n2 = min(n, len(arr2) - index2)
    n_eff = max(0, min(n1, n2))

    # Dot product: sum of (arr1[index1+i] * arr2[index2+i]) for i in 0..n-1
    result = np.int64(0)
    for i in range(n_eff):
        val1 = np.int64(arr1[index1 + i])
        val2 = np.int64(arr2[index2 + i])
        result += val1 * val2

    out_frac = np.array([np.uint8(frac1 + frac2)], dtype=np.uint8)
    result_out = np.array([result], dtype=np.int64)

    return out_frac, result_out


examples = generate_examples(inputs_info=inputs_info)
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)
pc.write_json("kernel.json")
