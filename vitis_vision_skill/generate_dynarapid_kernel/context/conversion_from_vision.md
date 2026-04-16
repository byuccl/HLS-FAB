# Converting Vitis Vision Kernels to Dynarapid Format

This guide explains how to convert a `generate_vision_kernel` example to a `generate_dynarapid_kernel` example. The key difference is the **container type approach**: the function signature uses `int32_t*` or `uint32_t*`, the `inputs_info` dtype matches that container type, and the behavioral function casts to/from the original Vitis pixel types internally.

## The Container Pattern

**Dynarapid requires:** `int32_t*` or `uint32_t*` for all pointer parameters (uniform pointer types)  
**Vitis Vision uses:** Various types (`ap_uint<8>*`, `ap_uint<16>*`, `ap_uint<32>*`, etc.)

The solution: use `int32_t*` or `uint32_t*` container types in both the function signature and `inputs_info`, while the behavioral function casts to the original pixel types for processing.

---

## Step-by-Step Conversion

### 1. Function Signature: Change `ap_uint<N>*` to `int32_t*` or `uint32_t*`

**Before (vision kernel):**
```python
function_signature = (
    "void boxfilter_accel("
    "ap_uint<8>* img_inp, "
    "ap_uint<8>* img_out)"
)
```

**After (dynarapid kernel):**
```python
function_signature = (
    "void boxfilter_accel("
    "int32_t* img_inp, "
    "int32_t* img_out)"
)
# OR
function_signature = (
    "void boxfilter_accel("
    "uint32_t* img_inp, "
    "uint32_t* img_out)"
)
```

Replace all `ap_uint<N>*` with either `int32_t*` or `uint32_t*` (choose one and use it consistently). No other changes to the function name or parameter order.

---

### 2. inputs_info: Match Function Signature Container Type

The `dtype` field in `inputs_info` must match the function signature type: either `"int32"` or `"uint32"`.

**Before (vision kernel):**
```python
inputs_info = [
    {
        "name": "img_inp",
        "dtype": "uint8",      # Original Vitis type
        ...
    },
    {
        "name": "img_out",
        "dtype": "uint8",
        ...
    },
]
```

**After (dynarapid kernel — using int32_t signature):**
```python
inputs_info = [
    {
        "name": "img_inp",
        "dtype": "int32",      # Match the int32_t* signature
        ...
    },
    {
        "name": "img_out",
        "dtype": "int32",      # Match the int32_t* signature
        ...
    },
]
```

**OR (dynarapid kernel — using uint32_t signature):**
```python
inputs_info = [
    {
        "name": "img_inp",
        "dtype": "uint32",     # Match the uint32_t* signature
        ...
    },
    {
        "name": "img_out",
        "dtype": "uint32",     # Match the uint32_t* signature
        ...
    },
]
```

All other fields (`shape`, `total_elements`, `iotype`, `min`, `max`) remain identical.

---

### 3. Behavioral Function: Cast to/from Original Pixel Type

The behavioral function must cast from the container type (int32 or uint32) to the original pixel type for processing, then cast back to the container type for the output.

**Before (vision kernel):**
```python
def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.uint8)
    # ... rest of algorithm using uint8 ...
    return out.astype(np.uint8).flatten()
```

**After (dynarapid kernel):**
```python
def behavioral(img_inp):
    # Cast from container (int32/uint32) to original pixel type
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.uint8)
    # ... rest of algorithm using uint8 (same as vision kernel) ...
    # Cast back to container type if needed, or return as-is if numpy handles it
    return out.astype(np.uint8).flatten()
```

**The behavioral function logic is IDENTICAL to the vision kernel.** The casts are already there — they convert from the container type to the original pixel type for processing. No algorithm changes needed.

---

## Examples

### Example 1: boxfilter (uint8 → uint8)

| Aspect | Vision Kernel | Dynarapid Kernel |
|--------|---|---|
| **Signature** | `ap_uint<8>* img_inp, ap_uint<8>* img_out` | `int32_t* img_inp, int32_t* img_out` |
| **inputs_info dtype** | `"uint8"` | `"int32"` (matches signature) |
| **behavioral()** | `img.astype(np.uint8)` | `img.astype(np.uint8)` (SAME — casts from container to pixel type) |

### Example 2: sobelfilter (uint8 input, int16 outputs)

| Aspect | Vision Kernel | Dynarapid Kernel |
|--------|---|---|
| **Signature** | `ap_uint<8>* img_inp, ap_uint<16>* img_out1, ap_uint<16>* img_out2` | `int32_t* img_inp, int32_t* img_out1, int32_t* img_out2` |
| **inputs_info dtypes** | `"uint8"`, `"int16"`, `"int16"` | `"int32"`, `"int32"`, `"int32"` (all match signature) |
| **behavioral()** | `img.astype(np.float32)` then `gx_int16 = np.clip(...).astype(np.int16)` | IDENTICAL (casts to original types for processing) |

### Example 3: addweighted (two uint8 inputs, uint8 output)

| Aspect | Vision Kernel | Dynarapid Kernel |
|--------|---|---|
| **Signature** | `ap_uint<8>* img_in1, ap_uint<8>* img_in2, ap_uint<8>* img_out` | `int32_t* img_in1, int32_t* img_in2, int32_t* img_out` |
| **inputs_info dtypes** | `"uint8"`, `"uint8"`, `"uint8"` | `"int32"`, `"int32"`, `"int32"` (all match signature) |
| **behavioral()** | Takes two args, processes as uint8, returns uint8 | IDENTICAL (casts from container to pixel type for processing) |

---

## Key Takeaways

1. **Function signature changes**: `ap_uint<N>*` → `int32_t*` or `uint32_t*`
2. **inputs_info dtype matches the signature**: Use `"int32"` or `"uint32"` (whichever you chose for the signature)
3. **behavioral() logic is IDENTICAL**: The casts to original pixel types (e.g., `.astype(np.uint8)`) are already there
4. **Pixel values stay in their original range during processing**: uint8 pixels are 0–255, int16 pixels are -32768–32767; the container type is just the wrapper

---

## Checklist

When converting a vision kernel to dynarapid:

- [ ] Replace all `ap_uint<N>*` with `int32_t*` or `uint32_t*` in `function_signature` (use consistently)
- [ ] Change all `dtype` fields in `inputs_info` to match the container type (`"int32"` or `"uint32"`)
- [ ] Keep the `behavioral()` function identical (no logic changes)
- [ ] Verify the `description` string accurately reflects the algorithm (no mention of wrapper or container types)
- [ ] Test `behavioral()` by calling it with numpy arrays to ensure the casts work correctly

