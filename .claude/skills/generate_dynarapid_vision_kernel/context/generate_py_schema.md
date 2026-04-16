# generate.py Schema Reference (DynaRapid int32 Version)

This document describes every field and function in a valid DynaRapid `generate.py` for HLS-FAB.

## Full File Structure

```python
import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants ---
# Extracted directly from xf_config_params.h
HEIGHT = <int>
WIDTH = <int>
# Other relevant compile-time constants (FILTER_WIDTH, NEWHEIGHT, NEWWIDTH, etc.)

# --- Logical pixel range ---
# Original Vitis type: <XF_TYPE> → logical range [MIN, MAX]
# Values are stored in int32 but represent <original_type> pixels.
PIXEL_MIN = <min>
PIXEL_MAX = <max>

# --- Function Signature ---
# CRITICAL: DynaRapid ONLY accepts int32_t for all pointer parameters.
# Function name must have _int32 suffix.
# Do NOT include rows or cols — image dimensions are compile-time constants.
function_signature = "void <func_name>_int32(int32_t* img_inp, int32_t* img_out)"

# --- Description ---
# Precise English description of the kernel algorithm.
# MUST state the logical pixel range and that values are stored in int32.
# Do NOT mention Vitis, xf::cv, XF_BORDER_CONSTANT, or any external library.
description = f"""..."""

# --- inputs_info ---
# One dict per buffer parameter. ALL dtypes must be "int32".
# min/max must use the LOGICAL pixel range, not full int32 range.
inputs_info = [
    {
        "name": "<param_name>",           # Matches the C parameter name exactly
        "array_dimension": <1, 2, or 3>,
        "shape": (<dim1>, <dim2>, ...),
        "total_elements": <int>,
        "dtype": "int32",                 # ALWAYS int32 for DynaRapid
        "iotype": "in",
        "min": PIXEL_MIN,                 # Logical range, NOT -2**31
        "max": PIXEL_MAX,                 # Logical range, NOT 2**31-1
    },
    {
        "name": "<output_param_name>",
        "array_dimension": <1, 2, or 3>,
        "shape": (<dim1>, <dim2>, ...),
        "total_elements": <int>,
        "dtype": "int32",                 # ALWAYS int32 for DynaRapid
        "iotype": "out",
    },
]

# --- behavioral() ---
# Pure numpy reference implementation.
# MUST clamp output to logical pixel range before returning.
def behavioral(<in_param_1>, <in_param_2>, ...):
    img = <in_param_1>.reshape((<HEIGHT>, <WIDTH>))
    # ... perform computation (use int64 accumulators to avoid overflow) ...
    result = ...
    return np.clip(result, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()

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

## Key Differences from Vitis Version

| Aspect | Vitis Version | DynaRapid Version |
|---|---|---|
| Pointer types | `ap_uint<N>*` | `int32_t*` (always) |
| dtype in inputs_info | `"uint8"`, `"int16"`, etc. | `"int32"` (always) |
| min/max in inputs_info | Type's full range | **Logical pixel range** (e.g. 0-255) |
| Function name suffix | None | `_int32` |
| Output directory | `dataset/kernels/` (Vitis, no suffix) | `dataset/kernels/` (DynaRapid, `_int32` suffix) |
| behavioral() output | `.astype(np.uint8)` etc. | `np.clip(...).astype(np.int32)` |
| Float kernels | Supported | **SKIP** |

---

## inputs_info Field Reference

| Field | Type | Required | DynaRapid Notes |
|---|---|---|---|
| `name` | str | Yes | Must match the C parameter name |
| `array_dimension` | int | Yes | 1, 2, or 3 |
| `shape` | tuple | Yes | Logical shape, e.g. `(HEIGHT, WIDTH)` |
| `total_elements` | int | Yes | Must equal `math.prod(shape)` |
| `dtype` | str | Yes | **Always `"int32"`** |
| `iotype` | str | Yes | `"in"` or `"out"` |
| `min` | number | For "in" only | Logical pixel min (e.g. 0 for uint8) |
| `max` | number | For "in" only | Logical pixel max (e.g. 255 for uint8) |

---

## behavioral() Rules

1. **Inputs**: One positional argument per `"in"` entry in `inputs_info`, in order.
2. **Return**: A single **flat** `int32` numpy array (or tuple of flat arrays for multiple outputs).
3. **numpy only**: No scipy, cv2, or any library other than numpy.
4. **Reshape first**: Always reshape flat inputs using `HEIGHT`, `WIDTH` constants.
5. **Use int64 accumulators**: To avoid overflow in intermediate computations, cast to `int64` before arithmetic.
6. **Clamp output**: Always `np.clip(result, PIXEL_MIN, PIXEL_MAX).astype(np.int32)` before `.flatten()`.
7. **Match the C++ arithmetic**: Preserve the original kernel's division, rounding, and saturation behaviour.

---

## Common behavioral() Patterns (DynaRapid int32)

### Box/Mean filter
```python
PIXEL_MIN = 0
PIXEL_MAX = 255

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.int64)
    pad = FILTER_WIDTH // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros((HEIGHT, WIDTH), dtype=np.int64)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            window_sum = padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH].sum()
            out[i, j] = window_sum // (FILTER_WIDTH * FILTER_WIDTH)
    return np.clip(out, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()
```

### Sobel filter (two outputs)
```python
INPUT_PIXEL_MIN = 0
INPUT_PIXEL_MAX = 255
OUTPUT_PIXEL_MIN = -32768
OUTPUT_PIXEL_MAX = 32767

SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.int64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.int64)

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.int64)
    gx = _convolve2d(img, SOBEL_X)
    gy = _convolve2d(img, SOBEL_Y)
    gx_out = np.clip(gx, OUTPUT_PIXEL_MIN, OUTPUT_PIXEL_MAX).astype(np.int32)
    gy_out = np.clip(gy, OUTPUT_PIXEL_MIN, OUTPUT_PIXEL_MAX).astype(np.int32)
    return gx_out.flatten(), gy_out.flatten()
```

### Threshold
```python
PIXEL_MIN = 0
PIXEL_MAX = 255
THRESH = 128
MAXVAL = 255

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH))
    out = np.where(img > THRESH, MAXVAL, 0)
    return np.clip(out, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()
```

### Weighted blend (multiple inputs)
```python
PIXEL_MIN = 0
PIXEL_MAX = 255
ALPHA = 0.5
BETA = 0.5
GAMMA = 0.0

def behavioral(img_in1, img_in2):
    a = img_in1.reshape((HEIGHT, WIDTH)).astype(np.float64)
    b = img_in2.reshape((HEIGHT, WIDTH)).astype(np.float64)
    out = ALPHA * a + BETA * b + GAMMA
    return np.clip(out, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()
```
