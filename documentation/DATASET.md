# Dataset
This document is provided to layout the dataset that we have developed and to give links to the sources from which we gathered some kernels. 

## Current Kernels in Dataset

We have broken up our dataset into 3 sub datasets: Dynarapid, Vitis Tools, Vitis Library Comparison.
  **Dynarapid:**
    This subset is focused on the testing and benchmarking of the Dynarapid HLS tool. The Dynarapid framwork only supports 32 bit integers and as such every kernel in this sub dataset uses this datatype. 

| #  | Category                                          | Kernel Count | Kernels                   |
|----|---------------------------------------------------|--------------|---------------------------|
| 1  | Element Wise & Activation                         | 10           | `abs_int32`, `add_offset_int32`, `elementwise_max_int32`, `elementwise_min_int32`, `hardsigmoid_int32`, `negate_int32`, `relu_int32`, `sign_int32`, `vectoradd_int32`, `vectorsubtract_int32`  |
| 2  | Spatial / Linear-Algebra                          | 10           | `conv1d_int32`, `conv2d_int32`, `dotproduct_bias_relu_int32`, `dotproduct_int32`, `matmul_16x6_int32`, `matrixvector_mult_int32`, `maxpool1d_uint32`, `maxpool2d_relu_int32`, `scale_matrix_3d_int32`, `vectormatrix_mult_int32`  |
| 3  | Reductions & Statistics / Loss                    | 2            | `argmax_int32`, `reduce_add_relu_int32` |
| 4  | Bitwise, Comparison, Casting & Data Movement / Padding      | 10           | `bitcount_uint32`, `bitwisexor_uint32`, `compare_gt_int32`, `compare_lt_int32`, `pad1d_int32`, `pad2d_int32`, `maxpool1d_uint32`, `maxpool2d_relu_int32`, `scale_matrix_3d_int32`, `vectormatrix_mult_int32`  |

 **Vitis**
    This subset of kernels is focused on the comercial HLS tool, Vitis Unified IDE. This tool can take virtually any size or type of data so we test it with various datatypes and sizes (int32, bfloat16, uint16, etc).

| #  | Category                                          | Kernel Count | Kernels                   |
|----|---------------------------------------------------|--------------|---------------------------|
| 1  | Element Wise & Activation                         | 46           | `abs_int8`, `add_offset_int8`, `add_offset_uint8`, `ceil_bfloat16`, `complexabs_bfloat16`, `cos_bfloat16`, `divide_bfloat16`, `elementwise_max_bfloat16`, `elementwise_max_int8`, `elementwise_min_bfloat16`, `elementwise_min_int8`, `exp_bfloat16`, `floor_bfloat16`, `gelu_bfloat16`, `hardsigmoid_bfloat16`, `hardsigmoid_int8`, `hardswish_bfloat16`, `inverse_uint8`, `leaky_relu_bfloat16`, `log10_bfloat16`, `log2_bfloat16`, `log_bfloat16`, `mish_bfloat16`, `negate_bfloat16`, `negate_int8`, `reciprocal_bfloat16`, `relu6_bfloat16`, `relu_bfloat16`, `relu_bfloat16_cast_uint8`, `relu_int8`, `round_bfloat16`, `rsqrt_bfloat16`, `sigmoid_bfloat16`, `sign_int8`, `sin_bfloat16`, `softmax_bfloat16`, `sqrt_bfloat16`, `tan_bfloat16`, `tanh_bfloat16`, `vectoradd_bfloat16`, `vectoradd_int16`, `vectoradd_relu_bfloat16`, `vectormult_bfloat16`, `vectorsubtract_bfloat16`, `vectorsubtract_int8`  |
| 2  | Spatial / Linear-Algebra                          | 24           | `avgpool1d_bfloat16`, `avgpool1d_relu_bfloat16`, `avgpool2d_bfloat16`, `avgpool2d_relu_bfloat16`, `conv1d_bfloat16`, `conv1d_bias_relu_bfloat16`, `conv1d_int32`, `conv1d_k2_s1_bias_relu_bfloat16`, `conv1d_k2_s2_bias_int16`, `conv1d_k4_s1_bias_relu_bfloat16`, `conv2d_bfloat16`, `conv2d_int32`, `conv2d_k2_s1_bias_relu_bfloat16`, `conv2d_k4_s2_bias_relu_bfloat16`, `dotproduct_bfloat16`, `dotproduct_bias_relu_bfloat16`, `dotproduct_bias_relu_int8`, `dotproduct_int32`, `gather_bfloat16_int32idx`, `maxpool1d_uint8`, `maxpool2d_bfloat16`, `maxpool2d_relu_bfloat16`, `maxpool2d_relu_int8`, `vectormatrix_mult_int32`  |
| 3  | Reductions & Statistics / Loss                    | 10            | `argmax_bfloat16`, `argmax_int32`, `argmin_bfloat16`, `euclidean_dist_bfloat16`, `l1_norm_bfloat16`, `max_abs_bfloat16`, `mse_loss_bfloat16`, `reduce_add_relu_int8`, `reducemin_bfloat16`, `variance_bfloat16` |
| 4  | Bitwise, Comparison, Casting & Data Movement / Padding      | 17           | `bitcount_uint16`, `bitcount_uint8`, `bitwiseand_uint8`, `bitwisenot_uint8`, `bitwiseor_uint8`, `bitwisexor_uint8`, `cast_bfloat16_to_float32`, `cast_bfloat16_to_int8`, `cast_float32_to_bfloat16`, `cast_int8_to_int32`, `compare_equal_bfloat16`, `compare_equal_int32`, `compare_gt_int8`, `compare_lt_int8`, `pad1d_int32`, `pad2d_int32`  |

 **Vitis Libraries Comparison**
    AMD (Formerly Xilinx) has created their own library of accelerated kernels for use in different cases. We have taken a few of these kernels specifcally from the Vitis Vision Libraries and the Vitis BLAS(Linear Algebra) Libraries. 

| #  | Category                                          | Kernel Count | Kernels                   |
|----|---------------------------------------------------|--------------|---------------------------|
| 1  | Computer Vision / Images                          | 10           | `accumulate_squared_opencv, `rotate_opencv`, `sobel_filter_opencv`, `auto_exposure_correction_opencv`, `bilateral_filter_opencv`, `accumulate_weighted_opencv`, `accumulate_opencv`, `bit_depth_convert_opencv`, `add_weighted_opencv`, `reduce_avg_opencv`  |

| #  | Category                                          | Kernel Count | Kernels                                                                                             |
|----|---------------------------------------------------|--------------|------------------------------------------------------------------------------------------------------|
| 1  | Element-wise & Activation                         | 4            | `absolute_value_int8`, `subtract_float64`, `zero_mean_norm_float64`, `relu_activation_float32`       |
| 2  | Spatial / Linear-Algebra                          | 5            | `dot_product_float32`, `matrix_multiply_int32`, `affine_float32`, `conv2d_int16`, `scale_matrix_3d_int8` |
| 3  | Reductions & Statistics / Loss                    | 3            | `mean_float32`, `std_deviation_float32`, `mean_var_float64`                                          |
| 4  | Bitwise, Comparison, Casting & Data Movement / Padding | 0            | —                                                                                                    |
| 5  | Signal & Image Processing                         | 2            | `fir_filter_float32`, `sobel_edge_detect_uint8`                                                      |

## Categories for Kernels

### Element-wise & Activation
As the table shows these are the kernels that have either single inputs or pairs of operands. These inputs do not depend on spatial structures. Examples of these types of arithmetics are add, subtract, max/min, and exp/log.

### Spatial & Linear Algebra
These kernels are functions that use neighborhood or tensor layout, or anything with matrices. Some examples are convolutions, matrix operations.
### Reductions & Statistics
This category of kernel are all operations that collapse dimensions or compute aggregate stats or distances. Examples include means, MSE, variance.
### Bitwise, Comparison, Casting & Data movement / Padding
These kernels are low level operations logical or bitwise. They deal with popcount, equality/greater/less predicates, type-conversions, padding and shuffles.
### Signal & Image Processing
These are the kernels that deal with DSP or Image processing. Filtering, transformations, blurs, edge detection, etc. 

## generate_test.py
We have a script called `generate_tests.py` within the `dataset` folder. This script uses the example kernels in the `examples` folder to make an LLM call. The response of the LLM is 2 files which are the `canonical_scalar.cc` and `generate.py` for each new kernel. 
### Prompt for Generate Tests
```python
SYSTEM_PROMPT = """
You are a part of a benchmarking system for HLS (High Level Synthesis).
You are very skilled both in writing python and C++ code.
Your job is to create the base code for tests within the benchmark.
Each test should have a C++ implementation that hasn't been accellerated at all. This will serve as a functional example and provide a method to test HLS solutions.
While this C++ implementation will not be synthesized for HLS, you should avoid using techniques and libraries that would fail to synthesize.
Additionally, each test within the benchmark will have a Python file with a description of the task and functions to facilitate testing.
Both these files should follow the patterns set out by the provided examples.
Produce only the code requested, no main, no additional examples or code, and no wrappers of any kind.
"""

desired_test = "new_kernel_name"
desired_test_description = (
    "Description for what the new kernel does."
)
```

## Credit for kernels
[Vitis_Libraries](https://github.com/Xilinx/Vitis_Libraries)<br>
[HiFlipVX](https://github.com/TUD-ADS/HiFlipVX)<br>
[Edge Detector HW/SW Co-design on FPGA](https://github.com/rezaAdinepour/Edge-Detector-Co-Design)<br>
[HLS-Project](https://github.com/JohnBagshaw/HLS-Project)<br>