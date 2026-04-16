# Vitis Vision L1 Kernel Reference (DynaRapid Compatibility)

This file lists all 88 kernels in `${VITIS_LIBRARIES}/vision/L1/examples/`
and notes their DynaRapid compatibility.

## Kernel Directory Listing

```
3dlut                  distancetransform      magnitude
accumulate             erosion                meanshifttracking
accumulatesquared      fast                   meanstddev
accumulateweighted     flip                   medianblur
addweighted            gaincontrol            minmaxloc
aec                    gammacorrection        modefilter
arithm                 gaussiandifference     otsuthreshold
autogaincontrol        gaussianfilter         paintmask
autowhitebalance       globaltonemapping      phase
axiconv                harris                 pyrdown
badpixelcorrection     hdrdecompand           pyrup
bfmatcher              hdrextractfrms         quantizationdithering
bilateralfilter        hdrmerge               readme
blacklevel             histequalize           reduce
boundingbox            histogram              remap
boxfilter              hog                    resize
ccacustom              houghlines             rgbirbayer
channelcombine         inrange                rotate
channelextract         integralimg            scharrfilter
clahe                  isppipeline            sgbm
colorcorrectionmatrix  isppipeline_autogain   sobelfilter
colordetect            isppipeline_hdr        stereolbm
convertbitdepth        isppipeline_mono       stereopipeline
convertscaleabs        isppipeline-rgbir      sum
custom_bgr2y8          ispstats               svm
customconv             kalmanfilter           threshold
cvtcolor               lensshadingcorrection  tonemapping
degamma                lkdensepyrof           warptransform
demosaicing            lknpyroflow
dilation               lut
```

## DynaRapid Compatibility Notes

### Likely SKIP (floating-point types)

These kernels typically use `float`, `double`, `ap_fixed`, or `XF_32FC1` and should
be skipped for DynaRapid:

- `gammacorrection` — uses floating-point gamma computation
- `globaltonemapping` — floating-point tone mapping
- `tonemapping` — floating-point tone mapping
- `kalmanfilter` — floating-point Kalman state
- `lkdensepyrof` / `lknpyroflow` — optical flow with float computation
- `hdrmerge` / `hdrdecompand` / `hdrextractfrms` — HDR processing with float
- `warptransform` — floating-point transformation matrices
- `remap` — floating-point coordinate mapping
- `sgbm` / `stereolbm` / `stereopipeline` — stereo matching with float
- `svm` — support vector machine with float

Always check the actual `xf_config_params.h` — some kernels may have integer-only
configurations even if they commonly use float.

### Good Candidates (integer types only)

These kernels use `XF_8UC1`, `XF_8UC3`, `XF_16SC1`, `XF_16UC1`, or `XF_32SC1` and
are good candidates for DynaRapid int32 conversion:

- `boxfilter` — uint8 mean filter
- `sobelfilter` — uint8 input, int16 gradient output
- `scharrfilter` — uint8 input, int16 gradient output
- `gaussianfilter` — uint8 Gaussian blur (if integer coefficients)
- `medianblur` — uint8 median filter
- `threshold` — uint8 thresholding
- `addweighted` — uint8 weighted blend (uses float intermediates but int I/O)
- `dilation` / `erosion` — uint8 morphological ops
- `flip` — uint8 image flip
- `resize` — uint8 nearest-neighbour resize
- `accumulate` / `accumulatesquared` / `accumulateweighted` — accumulation ops
- `channelcombine` / `channelextract` — multi-channel ops
- `histogram` / `histequalize` — histogram operations
- `convertbitdepth` — bit depth conversion
- `inrange` — range thresholding
- `magnitude` / `phase` — gradient magnitude/phase (int16 input)
- `pyrdown` / `pyrup` — pyramid operations

## Common Source File Naming Pattern

For a kernel named `<kernel>`:
- Accel source:  `xf_<kernel>_accel.cpp`  (sometimes `xf_<abbreviated>_accel.cpp`)
- Accel config:  `xf_<kernel>_accel_config.h`
- Params config: `config/xf_config_params.h`
- Description:   `description.json`

## Standard Parameter Patterns

Most kernels follow this function signature pattern:
```cpp
// Original Vitis:
void <func>_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_inp,
    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
    int rows, int cols
);

// DynaRapid int32 conversion:
void <func>_accel_int32(
    int32_t* img_inp,
    int32_t* img_out
);
```

### Kernels with scalar parameters (non-exhaustive)
| Kernel | Extra scalar params |
|---|---|
| `threshold` | `thresh` (uchar), `maxval` (uchar) |
| `gaussianfilter` | `sigma` (float — check if int config exists) |
| `resize` | `rows_in`, `cols_in`, `rows_out`, `cols_out` |
| `bilateralfilter` | `filter_strength` |
| `convertscaleabs` | `alpha`, `beta` |

### Kernels with multiple outputs
| Kernel | Outputs |
|---|---|
| `sobelfilter` | `img_out1` (Gx), `img_out2` (Gy) |
| `scharrfilter` | `img_out1` (Gx), `img_out2` (Gy) |
| `meanstddev` | `mean`, `stddev` (scalar outputs) |

### Kernels where OUTPUT type differs from INPUT type
| Kernel | IN_TYPE | OUT_TYPE | Input Range | Output Range |
|---|---|---|---|---|
| `sobelfilter` | `XF_8UC1` | `XF_16SC1` | 0–255 | -32768–32767 |
| `scharrfilter` | `XF_8UC1` | `XF_16SC1` | 0–255 | -32768–32767 |
| `magnitude` | `XF_16SC1` | `XF_8UC1` | -32768–32767 | 0–255 |
| `convertbitdepth` | varies | varies | varies | varies |

## Output Location

Generated DynaRapid `generate.py` files go to:
```
${REPO_ROOT}/dataset/kernels/<top_function_name>_int32/generate.py
```

## Existing DynaRapid Kernels (non-vision, for reference)

These are in `${REPO_ROOT}/dataset/dynarapid_kernels/` and demonstrate the int32 pattern:
```
abs_int32, add_offset_int32, argmax_int32, bitcount_uint32,
bitwisexor_uint32, compare_gt_int32, compare_lt_int32, conv1d_int32,
conv2d_int32, cumsum_int32, dotproduct_bias_relu_int32, dotproduct_int32,
elementwise_max_int32, elementwise_min_int32, hardsigmoid_int32,
matmul_16x16_int32, matrixvector_mult_int32, maxpool1d_uint32,
maxpool2d_relu_int32, negate_int32, pad1d_int32, pad2d_int32,
reduce_add_relu_int32, relu_int32, scale_matrix_3d_int32, sign_int32,
swap_int32, vectoradd_int32, vectormatrix_mult_int32, vectorsubtract_int32
```
