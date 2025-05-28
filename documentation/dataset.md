# Dataset
This document is provided to layout the dataset that we have developed and to give links to the sources from which we gathered some kernels. 

## Current Kernels in Dataset

Kernels that we have:

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
Prompt for Generate Tests
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
[Vitis_Libraries](https://github.com/Xilinx/Vitis_Libraries)
[HiFlipVX](https://github.com/TUD-ADS/HiFlipVX)
[Edge Detector HW/SW Co-design on FPGA](https://github.com/rezaAdinepour/Edge-Detector-Co-Design)
[HLS-Project] (https://github.com/JohnBagshaw/HLS-Project)