# Quick Reference: Vision → Dynarapid Conversion

Use this checklist when converting a `generate_vision_kernel` example to `generate_dynarapid_kernel`.

---

## Conversion Checklist

### Function Signature
- [ ] Replace all `ap_uint<N>*` with `int32_t*` or `uint32_t*` (pick one, use consistently)
  - **Before:** `ap_uint<8>* img_inp, ap_uint<8>* img_out`
  - **After:** `int32_t* img_inp, int32_t* img_out` OR `uint32_t* img_inp, uint32_t* img_out`
- [ ] Keep parameter names and order the same
- [ ] Keep function name the same

### inputs_info
- [ ] Change all `dtype` fields to match the container type chosen for the signature
  - **If signature uses int32_t:** set dtype to `"int32"`
  - **If signature uses uint32_t:** set dtype to `"uint32"`
- [ ] Example:
  ```python
  {
      "name": "img_inp",
      "dtype": "int32",      # Matches int32_t* signature
      "shape": (HEIGHT, WIDTH),
      "total_elements": HEIGHT * WIDTH,
      ...
  }
  ```
- [ ] Keep `shape`, `total_elements`, `iotype`, `min`, `max` unchanged

### behavioral() Function
- [ ] **NO ALGORITHM LOGIC CHANGES** — the casts are already there
- [ ] The cast to original dtype (e.g., `.astype(np.uint8)`) converts from container type
- [ ] Example:
  ```python
  def behavioral(img_inp):
      # Cast from container (int32/uint32) to original pixel type for processing
      img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.uint8)
      # ... rest of algorithm unchanged ...
      return out.astype(np.uint8).flatten()
  ```

### Description
- [ ] Update to mention **algorithm category**
  - Stencil-type (box filter, Sobel, etc.)
  - Histogram-based (equalization, etc.)
  - Reduction (sum, mean, etc.)
  - Multi-pass (Canny, etc.)
  - Variable-iteration (connected components, etc.)
- [ ] Note the **code structure and why it suits Dynamatic**
  - Example for stencil: "Flat loop nests with zero-padded border handling enable systolic-like dataflow"
  - Example for histogram: "Two-phase structure (accumulation → lookup) separates data-dependent indirection"
  - See `dynamatic_architecture_patterns.md` for templates

### Data Types (Validation)
- [ ] Verify no floating-point types (XF_32FC1, ap_fixed, ap_ufixed, float, double)
  - If yes: **SKIP** with reason "Floating-point type not supported by Dynarapid"
- [ ] Verify algorithm structure is Dynamatic-friendly (see `dynamatic_architecture_patterns.md`)
  - If complex pointer-chasing or dynamic allocation: **SKIP**

---

## Side-by-Side Comparison Template

| Aspect | Vision Kernel | Dynarapid Kernel |
|--------|---|---|
| Signature | `ap_uint<8>* img_inp, ap_uint<8>* img_out` | `int32_t* img_inp, int32_t* img_out` |
| inputs_info dtype | `"uint8"` | `"int32"` (matches container type) |
| behavioral() | `img.astype(np.uint8)` | `img.astype(np.uint8)` (same — casts from container) |
| Description | Generic algorithm description | Add algorithm category + Dynamatic structure note |

---

## Common Algorithm Patterns (Pick One)

### Stencil-Type (Box Filter, Sobel, Convolution)
```python
# Copy from vision kernel, update description:
description = f"""
...[original description]...

Algorithm category: Stencil-type operation

Structure: Flat loop nests with zero-padded border handling.
Each output pixel is computed from a neighborhood with zero-padding
to handle edges uniformly. This structure enables Dynamatic to create
systolic-like dataflow pipelines.
"""
```

### Histogram-Based (Histogram Equalization, CLAHE)
```python
# Copy behavioral from vision kernel, update description:
description = f"""
...[original description]...

Algorithm category: Histogram-based operation

Structure: Two-phase structure separates accumulation (histogram binning)
from lookup (LUT application). This separation allows Dynamatic's Load-Store
Queue to efficiently handle data-dependent bin indexing in the accumulation
phase without artificial constraints.
"""
```

### Reduction (Sum, Mean, Min/Max, Stddev)
```python
# Copy behavioral from vision kernel, update description:
description = f"""
...[original description]...

Algorithm category: Reduction operation

Structure: Partial reductions across rows (or tiles) are computed in the first phase,
then merged in a second reduction pass. This enables Dynamatic to parallelize
the partial reduction phase while keeping the final merge sequential.
"""
```

### Element-Wise (Add Weighted, Threshold)
```python
# Copy from vision kernel, update description:
description = f"""
...[original description]...

Algorithm category: Element-wise operation

Structure: Simple per-pixel computation with no neighborhood access.
This enables Dynamatic to fully parallelize across all pixels.
"""
```

---

## When to Skip

1. **Floating-point types**
   - Check: Does the Vitis type contain `F` (e.g., `XF_32FC1`, `ap_fixed`)?
   - Action: SKIP with reason "Floating-point type not supported by Dynarapid"

2. **Complex algorithms**
   - Check: Does the behavioral model use pointer-chasing, arbitrary graphs, or recursive calls?
   - Action: SKIP with reason "Algorithm structure not suitable for Dynamatic dataflow synthesis"

---

## Verification

After conversion, verify:
- [ ] Function signature has `int32_t*` or `uint32_t*` only
- [ ] inputs_info dtype matches the function signature container type (`"int32"` or `"uint32"`)
- [ ] behavioral() casts to original Vitis pixel type (e.g., `.astype(np.uint8)`)
- [ ] Description mentions algorithm category and structure
- [ ] No floating-point types
- [ ] Example data can be generated without errors

---

## Full Details

- **For detailed conversion examples:** See `conversion_from_vision.md`
- **For algorithm-specific structures:** See `dynamatic_architecture_patterns.md`
- **For type mapping reference:** See `xf_type_mapping.md`
- **For process and rules:** See `SKILL.md`

