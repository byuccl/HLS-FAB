# Dataset
    This document is provided to layout the dataset that we have developed and to give links to the sources from which we gathered some kernels. 

## Current Kernels in Dataset

Kernels that we have:
| # | Category                          | Kernel Count | Kernels |
|---|-----------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------|--------------|---------|
| 1 | Element-wise & Activation         | 4           | `absolute_value_int8`, `subtract_float64`, `zero_mean_norm_float64`, `relu_activation_float32`|
| 2 | Spatial / Linear-Algebra          | 5           | `dot_product_float32`, `matrix_multiply_int32`, `affine_float32`, `conv2d_int16`, `scale_matrix_3d_int8` |
| 3 | Reductions & Statistics / Loss    | 3           | `mean_float32`, `std_deviation_float32`, `mean_var_float64`,  |
| 4 | Bitwise, Comparison, Casting & Data movement / Padding | 0           | |
| 5 | Signal & Image Processing         | 2           | `fir_filter_float32`, `sobel_edge_detect_uint8` |

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