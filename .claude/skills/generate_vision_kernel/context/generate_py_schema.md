# generate.py Schema Reference

This document describes every field and function in a valid `generate.py` for HLS-FAB.

## Full File Structure

```python
import numpy as np
# Optional: from scipy.ndimage import ..., import cv2, from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

# --- Constants ---
# Extracted directly from xf_config_params.h
HEIGHT = <int>
WIDTH = <int>
# Other relevant compile-time constants (FILTER_WIDTH, NEWHEIGHT, NEWWIDTH, etc.)

# --- Function Signature ---
# Must exactly match the C/C++ HLS function declaration.
# Use ap_uint<N> where N = INPUT_PTR_WIDTH / OUTPUT_PTR_WIDTH from xf_config_params.h
# Do NOT include rows or cols — image dimensions are compile-time constants (HEIGHT, WIDTH).
function_signature = "void <func_name>(ap_uint<N>* img_inp, ap_uint<M>* img_out)"

# --- Description ---
# A long-form, precise English description of what the kernel does.
# IMPORTANT: Do NOT mention Vitis, xf::cv, XF_BORDER_CONSTANT, or any external
# library. Describe the algorithm in plain terms only.
# e.g. say "zero-padding" not "XF_BORDER_CONSTANT", "box filter" not "xf::cv::boxFilter".
description = f"""..."""

# --- inputs_info ---
# A list of dicts, one per buffer parameter (inputs AND outputs).
# Scalar params (rows, cols, thresh, sigma, etc.) are NOT listed here.
# Order must match the C function parameter order.
inputs_info = [
    {
        "name": "<param_name>",           # Matches the C parameter name exactly
        "array_dimension": <1, 2, or 3>,  # Number of logical dimensions
        "shape": (<dim1>, <dim2>, ...),    # Logical shape (not ap_uint word count)
        "total_elements": <int>,           # Product of all shape dimensions
        "dtype": "<numpy_dtype_string>",   # e.g. "uint8", "int16", "float32"
        "iotype": "in",                    # "in" for inputs
        "min": <number>,                   # Min value for random test generation
        "max": <number>,                   # Max value for random test generation
    },
    {
        "name": "<output_param_name>",
        "array_dimension": <1, 2, or 3>,
        "shape": (<dim1>, <dim2>, ...),
        "total_elements": <int>,
        "dtype": "<numpy_dtype_string>",
        "iotype": "out",                   # "out" for outputs (no min/max needed)
    },
]

# --- behavioral() ---
# Pure Python/numpy reference implementation. 
# Arguments: one per "in" entry in inputs_info, in the same order, as flat numpy arrays.
# Returns: single flat numpy array matching the "out" entry's dtype and total_elements.
def behavioral(<in_param_1>, <in_param_2>, ...):
    # Reshape inputs to their logical shapes
    img = <in_param_1>.reshape((<HEIGHT>, <WIDTH>))
    # ... perform computation ...
    result = ...
    return result.astype(np.<dtype>).flatten()

# --- Generate examples and PromptConstructor ---
examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
)

pc.write_json("kernel.json")
```

---

## inputs_info Field Reference

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | str | Yes | Must match the C parameter name |
| `array_dimension` | int | Yes | 1, 2, or 3 |
| `shape` | tuple | Yes | Logical shape, e.g. `(HEIGHT, WIDTH)` |
| `total_elements` | int | Yes | Must equal `math.prod(shape)` |
| `dtype` | str | Yes | numpy dtype string: `"uint8"`, `"int16"`, `"float32"`, etc. |
| `iotype` | str | Yes | `"in"` or `"out"` |
| `min` | number | For "in" only | Minimum value for random data generation |
| `max` | number | For "in" only | Maximum value for random data generation |

---

## Description Precision Checklist

The `description` string must be specific enough that a programmer could reproduce `behavioral()` exactly from it alone. Before writing the description, answer each item by reading the C++ source, then include the answer in the description text:

| # | Question | Example of a vague answer (bad) | Example of a precise answer (good) |
|---|---|---|---|
| 1 | **How is the output value computed from the window?** | "average of the window" | "sum of all window pixels divided by FILTER_WIDTH², result truncated (floor) to uint8" |
| 2 | **What happens at image borders?** | "zero-padding" | "zero-padding; border pixels are still divided by the full FILTER_WIDTH² (9 for a 3×3 kernel), not by the count of in-bounds pixels" |
| 3 | **For weighted filters: what are the exact coefficients?** | "Gaussian weights" | "Gaussian kernel with coefficients [1,2,1; 2,4,2; 1,2,1] (sum=16), result right-shifted by 4 bits (divide by 16), truncated" |
| 4 | **How is the output stored / what is the output range?** | "stored as uint8" | "stored as uint8; values computed as int16 are saturated to [0,255] before storing" |
| 5 | **Are there any per-pixel scalar parameters (threshold, max value)?** | (omitted) | "pixels above THRESH are set to MAXVAL (255); pixels at or below THRESH are set to 0" |

If the C++ source is ambiguous about rounding, default to **truncation (floor cast)** — that is the most common pattern in HLS fixed-point arithmetic — and note it explicitly.

---

## behavioral() Rules

1. **Inputs**: One positional argument per `"in"` entry in `inputs_info`, in order.
2. **Return**: A single **flat** numpy array matching the `"out"` entry.
3. **numpy only**: No scipy, cv2, or any library other than numpy. Implement filters manually.
4. **Reshape first**: Always reshape flat inputs using `HEIGHT`, `WIDTH` constants.
5. **Cast output**: Always `.astype(np.<dtype>)` before `.flatten()`.
6. **Scalar parameters**: Hardcode them using the compile-time constants (e.g., `FILTER_WIDTH`).
7. **Match the C++ arithmetic**: Do not use Python/numpy float semantics unless the C++ source explicitly uses float. For integer kernels, use integer arithmetic to match the hardware exactly.

---

## Common behavioral() Patterns

**Rule**: Use numpy only. No scipy, no cv2. Implement filters with `np.pad` + manual convolution.

### Box/Mean filter (pure numpy)
```python
def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.float32)
    pad = FILTER_WIDTH // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros_like(img)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            out[i, j] = padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH].mean()
    return out.astype(np.uint8).flatten()
```

### Sobel filter (pure numpy, two outputs -> concatenate)
```python
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

def _convolve2d(img, kernel):
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode="constant", constant_values=0)
    out = np.zeros_like(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            out[i, j] = (padded[i:i+kh, j:j+kw] * kernel).sum()
    return out

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.float32)
    gx = _convolve2d(img, SOBEL_X)
    gy = _convolve2d(img, SOBEL_Y)
    gx = np.clip(gx, -32768, 32767).astype(np.int16)
    gy = np.clip(gy, -32768, 32767).astype(np.int16)
    return np.concatenate([gx.flatten(), gy.flatten()])
```

### Threshold (pure numpy)
```python
THRESH = 128
MAXVAL = 255
def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH))
    out = np.where(img > THRESH, MAXVAL, 0).astype(np.uint8)
    return out.flatten()
```

### Resize — nearest neighbour (pure numpy)
```python
NEWHEIGHT = 64
NEWWIDTH = 64
def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH))
    row_idx = (np.arange(NEWHEIGHT) * HEIGHT / NEWHEIGHT).astype(int)
    col_idx = (np.arange(NEWWIDTH) * WIDTH / NEWWIDTH).astype(int)
    out = img[np.ix_(row_idx, col_idx)]
    return out.astype(np.uint8).flatten()
```

### Gaussian filter (pure numpy)
```python
def _gaussian_kernel(size, sigma):
    ax = np.arange(size) - size // 2
    kernel = np.exp(-ax**2 / (2 * sigma**2))
    kernel2d = np.outer(kernel, kernel)
    return kernel2d / kernel2d.sum()

SIGMA = 1.0

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.float32)
    kernel = _gaussian_kernel(FILTER_WIDTH, SIGMA).astype(np.float32)
    pad = FILTER_WIDTH // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros_like(img)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            out[i, j] = (padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH] * kernel).sum()
    return out.astype(np.uint8).flatten()
```
