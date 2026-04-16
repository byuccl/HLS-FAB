---
name: generate_dynarapid_kernel
description: Generates a generate.py file for one, a listed subset, or all Dynarapid Vision Library L1 kernels, compatible with the HLS-FAB dataset format.
---

# Generate Dynarapid Kernel Skill

Generate HLS-FAB-compatible `generate.py` files for Vitis Vision L1 kernels optimized for Dynamatic synthesis via Dynarapid.

**Location:** `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/`  
**Output:** `/home/hoodoo17/skills_test/vision_kernels/<kernel_name>/generate.py`

---

## Input

One of:
- **Single kernel name:** `boxfilter`, `sobelfilter`, `gaussianfilter`, etc.
- **Batch mode:** `all` (all kernels) or `@/path/to/file.txt` (kernel list, one per line)

---

## Quick Start

### 1. Understand the Container Type Approach

Dynarapid requires `int32_t*` or `uint32_t*` pointers. Vision kernels use `ap_uint<N>*`. The conversion:

- **Function signature:** `ap_uint<N>*` → `int32_t*` or `uint32_t*`
- **inputs_info dtype:** matches container type (`"int32"` or `"uint32"`)
- **behavioral():** casts to original pixel type internally

**→ See `context/QUICK_REFERENCE.md` for a one-page conversion checklist**

---

### 2. Check if the Kernel is Suitable

**Skip if:**
- Uses floating-point types (`XF_32FC1`, `ap_fixed`, `ap_ufixed`, `float`, `double`)
- Has pointer-chasing or arbitrary dynamic structures
- Doesn't fit Dynamatic's dataflow model

**Likely suitable:**
- Stencil ops (box filter, Sobel, Gaussian, morphological ops)
- Histogram-based (histogram equalization, CLAHE)
- Simple reductions (sum, mean, min/max)
- Multi-pass with clear boundaries

---

### 3. Read the Source Files

For `<kernel>`, locate:
- `xf_<kernel>_accel.cpp` — top function implementation
- `config/xf_config_params.h` — `HEIGHT`, `WIDTH`, `IN_TYPE`, `OUT_TYPE`, `FILTER_WIDTH`, etc.
- `description.json` — `topfunction` name

---

### 4. Create generate.py

Use the conversion approach from `context/QUICK_REFERENCE.md` or `context/conversion_from_vision.md`:

```python
import numpy as np
from hlsfab import PromptConstructor, generate_examples

HEIGHT = <from config>
WIDTH = <from config>

# Container type: int32_t or uint32_t (choose one, be consistent)
function_signature = "void <top_func>(int32_t* img_inp, int32_t* img_out)"

description = """
<Algorithm description, mentioning category and Dynamatic structure.
See context/dynamatic_architecture_patterns.md for templates.>
"""

# dtype must match function signature container type
inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",      # Matches int32_t* signature
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",
        "dtype": "int32",      # Matches int32_t* signature
        "iotype": "out",
        ...
    },
]

def behavioral(img_inp):
    # Cast from container to original pixel type
    img = img_inp.reshape((HEIGHT, WIDTH)).astype(np.uint8)
    # ... algorithm logic (identical to vision kernel) ...
    return out.astype(np.uint8).flatten()

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

### 5. Verify and Describe

- Run behavioral() against test data to verify correctness
- Update description to note:
  - Algorithm category (stencil, histogram, reduction, multi-pass, variable-iteration)
  - Why the code structure suits Dynamatic synthesis

**→ See `context/dynamatic_architecture_patterns.md` for algorithm-specific guidance**

---

## Context Files

Quick reference and detailed guidance (read as needed):

| File | Purpose |
|------|---------|
| `context/QUICK_REFERENCE.md` | **One-page conversion checklist** — start here |
| `context/conversion_from_vision.md` | Detailed vision → dynarapid conversion with examples |
| `context/dynamatic_architecture_patterns.md` | Algorithm categories and structure patterns for Dynamatic |
| `context/xf_type_mapping.md` | Container type reference |
| `context/vitis_vision_kernels.md` | List of all 88 Vitis Vision kernels |

---

## Key Rules

- ⚠️ **int32_t or uint32_t ONLY** for all pointers
- ⚠️ **inputs_info dtype must match function signature** container type
- ⚠️ **Skip floating-point kernels** (XF_32FC1, ap_fixed, etc.)
- ⚠️ **No `rows`, `cols`** in function signature — use `HEIGHT`, `WIDTH` constants
- ✅ **behavioral() logic is unchanged** — casts handle container type internally
- ✅ **Structure code for Dynamatic** — use clear, explicit data flow patterns

---

## Examples in Skill Directory

See `examples/` for worked examples:
- `boxfilter/` — stencil-type, one input/output
- `addweighted/` — element-wise, two inputs
- `sobelfilter/` — stencil-type, multiple outputs

---

## Output Location

```
/home/hoodoo17/skills_test/vision_kernels/<top_function_name>/generate.py
```

Create the directory if it doesn't exist. The `<top_function_name>` is the C++ function name (e.g., `boxfilter_accel`, not `boxfilter`).

---

## Troubleshooting

**"Floating-point type not supported"?**  
→ Skip the kernel; Dynarapid doesn't support float/double/ap_fixed types.

**"inputs_info dtype mismatch"?**  
→ Ensure `inputs_info[*].dtype` matches the function signature (`int32` for `int32_t*`, `uint32` for `uint32_t*`).

**"behavioral() doesn't match C simulation"?**  
→ Double-check that casts match the original Vitis types. See the casts at the start of behavioral() — they should unwrap the container type to the original pixel type.

