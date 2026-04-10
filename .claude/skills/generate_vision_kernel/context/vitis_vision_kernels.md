# Vitis Vision L1 Kernel Reference

This file lists all 88 kernels in `${VITIS_LIBRARIES}/vision/L1/examples/`
and provides key configuration notes for each.

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

## Common Source File Naming Pattern

For a kernel named `<kernel>`:
- Accel source:  `xf_<kernel>_accel.cpp`  (sometimes `xf_<abbreviated>_accel.cpp`)
- Accel config:  `xf_<kernel>_accel_config.h`
- Params config: `config/xf_config_params.h`
- Description:   `description.json`

## Standard Parameter Patterns

Most kernels follow this function signature pattern:
```cpp
void <func>_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_inp,   // input image
    ap_uint<OUTPUT_PTR_WIDTH>* img_out,  // output image
    int rows,                             // runtime image height
    int cols                              // runtime image width
);
```

### Kernels with scalar parameters (non-exhaustive)
| Kernel | Extra scalar params |
|---|---|
| `threshold` | `thresh` (uchar), `maxval` (uchar) |
| `gaussianfilter` | `sigma` (float) |
| `resize` | `rows_in`, `cols_in`, `rows_out`, `cols_out` |
| `bilateralfilter` | `filter_strength` |
| `convertscaleabs` | `alpha`, `beta` |

### Kernels with multiple outputs
| Kernel | Outputs |
|---|---|
| `sobelfilter` | `img_out1` (Gx), `img_out2` (Gy) |
| `scharrfilter` | `img_out1` (Gx), `img_out2` (Gy) |
| `magnitude` / `phase` | `img_out` (magnitude or phase) |
| `meanstddev` | `mean`, `stddev` (scalar outputs) |

### Kernels where OUTPUT type differs from INPUT type
| Kernel | IN_TYPE | OUT_TYPE |
|---|---|---|
| `sobelfilter` | `XF_8UC1` | `XF_16SC1` |
| `scharrfilter` | `XF_8UC1` | `XF_16SC1` |
| `magnitude` | `XF_16SC1` | `XF_8UC1` |
| `convertbitdepth` | varies | varies |

## Output Location

Generated `generate.py` files go to:
```
dataset/kernels/<kernel_name>/generate.py
```
(relative to the HLS-FAB project root)

If the directory does not exist, create it first.

## Existing HLS-FAB Kernels (non-vision, for reference)

These are in `dataset/vitis_kernels/` (not Vitis Vision):
```
abs_int8, add_offset_int8, argmax_bfloat16, avgpool1d_bfloat16,
bitcount_uint8, bitwiseand_uint8, cast_bfloat16_to_float32,
conv1d_bfloat16, conv2d_bfloat16, matmul_16x16_int8, relu_int8,
vectoradd_bfloat16, ... (90+ kernels total)
```
