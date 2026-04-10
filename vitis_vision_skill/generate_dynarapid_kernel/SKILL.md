---
name: generate_dynarapid_vision_kernel
description: Generates a generate.py file for one, a listed subset, or all Dynarapid Vision Library L1 kernels, compatible with the HLS-FAB dataset format.
---
# Generate Dynarapid Vision Kernel Skill

Use this skill to generate a `generate.py` file for one or more kernels from the Xilinx Vitis Vision Library located at `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/` for compilation with Dynarapid HLS tools.

Each `generate.py` is part of the HLS-FAB dataset and describes a kernel's function signature, I/O buffers, behavioral Python model, and example data for AI-assisted HLS code generation.

---

## Input

The user will provide one of:

- A **single kernel name** — a directory name under `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/`. For example:
  - `boxfilter`
  - `sobelfilter`
  - `gaussianfilter`
  - `resize`
  - `threshold`

- The keyword **`all`** — to generate `generate.py` for every kernel directory found under `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/`, skipping any kernel whose output file already exists at `/home/hoodoo17/skills_test/vision_kernels/<kernel_name>/generate.py`.

- A path to a **text file** (e.g. `@/home/hoodoo17/skills_test/kernels_todo.txt`) — to generate `generate.py` for each kernel listed in the file, one kernel directory name per line. Lines starting with `#` and blank lines are ignored. Kernels whose output file already exists are skipped with `[SKIP]`.

  Example file contents:
  ```
  # kernels still to generate
  boxfilter
  sobelfilter
  gaussianfilter
  ```

---

## Batch Mode (`all` or `@<file>`)

When the input is `all`:

1. List every subdirectory of `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/` (excluding `readme` and any non-kernel directories) to build the kernel list.

When the input is `@<filepath>`:

1. Read the file at `<filepath>`. Split on newlines; strip each line; discard lines that are blank or start with `#`. The remaining lines are the kernel list.

For both modes, then for each kernel in the list:

   a. Read the kernel's `description.json` (or accel `.cpp`) to determine the **top function name**. Check if `/home/hoodoo17/skills_test/vision_kernels/<top_function_name>/generate.py` already exists. If it does, skip it and log `[SKIP] <top_function_name>`.
   b. Otherwise, run the full single-kernel process (Steps 1–8 below) for that kernel.
   c. Log `[OK] <kernel_name>` on success, or `[FAIL] <kernel_name>: <reason>` on failure — then continue to the next kernel without stopping.

After all kernels are processed, print a summary: total processed, skipped, succeeded, and failed.

---

## Process

> ⚠️ **CRITICAL — read before writing any file:**
> - **Dynarapid ONLY accepts int32_t types.** All pointer parameters in `function_signature` must be `int32_t*`. Skip any kernel using floating-point types (`float`, `double`, `ap_fixed`, `ap_ufixed`).
> - The `function_signature` must contain **only** the buffer pointer parameters (`int32_t*`). **Never include `rows`, `cols`, or any other runtime dimension parameter** — image dimensions are fixed compile-time constants (`HEIGHT`, `WIDTH`).
> - The `description` must **not** mention `rows` or `cols` as parameters.
> - The Vitis Vision C++ source uses `rows`/`cols` as runtime args, but the HLS-FAB format does not. Ignore them.

Follow these steps for every kernel:

### Step 1 — Read the kernel source files

For a kernel named `<kernel>`, read all of the following files (they may not all exist, use what is available):
- `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/<kernel>/xf_*_accel.cpp` — the top-level HLS kernel implementation
- `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/<kernel>/xf_*_accel_config.h` — main config header (includes `xf_config_params.h`)
- `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/<kernel>/config/xf_config_params.h` — defines `HEIGHT`, `WIDTH`, `IN_TYPE`, `OUT_TYPE`, `INPUT_PTR_WIDTH`, `OUTPUT_PTR_WIDTH`, `NPPCX`, `FILTER_WIDTH`, etc.
- `/home/hoodoo17/Vitis_Libraries/vision/L1/examples/<kernel>/description.json` — contains `topfunction` and other metadata

### Step 2 — Extract key parameters

From the source files, identify:

| Parameter | Where to find it |
|---|---|
| Top function name | `description.json` → `topfunction`, or the `void <name>(...)` in the accel `.cpp` |
| `HEIGHT`, `WIDTH` | `xf_config_params.h` |
| `IN_TYPE`, `OUT_TYPE` | `xf_config_params.h` (e.g. `XF_8UC1`, `XF_16SC1`) |
| `INPUT_PTR_WIDTH`, `OUTPUT_PTR_WIDTH` | `xf_config_params.h` — read for reference only; **do NOT use in the function signature** (see Step 3) |
| `NPPCX` | `xf_config_params.h` |
| Scalar parameters | Additional `int`, `unsigned char`, `float`, etc. arguments in the function signature (e.g. `thresh`, `maxval`, `sigma`) |
| Filter width | `FILTER_WIDTH` or similar defines |

### Step 3 — Determine numpy dtype and convert to int32_t

Dynarapid ONLY accepts `int32_t` for all pointer parameters. Convert all Vitis `XF_*` types to `int32_t` according to the mapping below. **Skip any kernel using floating-point types** (`XF_32FC1`, `XF_64FC1`, `ap_fixed`, `ap_ufixed`, etc.):

| Vitis Type | Convert to int32_t | numpy dtype | channels | Notes |
|---|---|---|---|---|
| `XF_8UC1` | `int32_t` | `int32` | 1 | Grayscale — convert uint8 to int32 |
| `XF_8UC3` | `int32_t` | `int32` | 3 | RGB/BGR 3-channel — convert uint8 to int32 |
| `XF_16UC1` | `int32_t` | `int32` | 1 | 16-bit unsigned — convert uint16 to int32 |
| `XF_16SC1` | `int32_t` | `int32` | 1 | 16-bit signed (gradients) — convert int16 to int32 |
| `XF_32SC1` | `int32_t` | `int32` | 1 | 32-bit signed — no conversion needed |
| `XF_32FC1` | **SKIP** | — | — | Floating-point — not supported by Dynarapid |
| `ap_fixed`, `ap_ufixed` | **SKIP** | — | — | Arbitrary-precision fixed-point — not supported by Dynarapid |
| Other float types | **SKIP** | — | — | Any `float`, `double`, `ap_fixed` — not supported by Dynarapid |

> ⚠️ **CRITICAL — int32_t ONLY.**
>
> **Dynarapid accepts ONLY `int32_t` for all pointer parameters.** 
> - Do NOT use `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `uint32_t`, `int64_t`, `uint64_t`, or any other C integer type.
> - Do NOT use `ap_int.h`, `ap_uint<N>`, `ap_fixed`, or `ap_ufixed` types.
> - Do NOT use any floating-point types (`float`, `double`, `ap_fixed`, `ap_ufixed`).
> - Always convert all Vitis `XF_*` types to `int32_t`, regardless of the source width or signedness.
> - If a kernel uses `float`, `double`, or arbitrary-precision fixed-point, **SKIP IT** with reason "Floating-point type not supported by Dynarapid".
>
> `INPUT_PTR_WIDTH` is a hardware AXI bus-width optimization (often 64, 128, or 256) that controls how many pixels are transferred per clock cycle. It does **not** change the pointer type. Always use `int32_t*`.
>
> Examples of correct mappings (ALL use int32_t):
> - `IN_TYPE = XF_8UC1`, `INPUT_PTR_WIDTH = 128` → use **`int32_t*`**, `dtype: "int32"`
> - `IN_TYPE = XF_16SC1`, `INPUT_PTR_WIDTH = 128` → use **`int32_t*`**, `dtype: "int32"`
> - `IN_TYPE = XF_8UC3`, `INPUT_PTR_WIDTH = 64` → use **`int32_t*`**, `dtype: "int32"`, `shape: (H, W, 3)`

For `inputs_info`, use the logical pixel count as `total_elements = HEIGHT * WIDTH` (or `HEIGHT * WIDTH * channels` for multi-channel), and treat `shape` as `(HEIGHT, WIDTH)` or `(HEIGHT, WIDTH, channels)`.

The `dtype` in `inputs_info` must be the numpy type matching the XF pixel type (e.g. `uint8`, `int16`, `uint16`), **not** the Vitis type, and **not** derived from `INPUT_PTR_WIDTH`.

### Step 4 — Identify scalar (non-buffer) parameters

Unlike the Vitis Vision C++ source, the HLS-FAB `generate.py` treats image dimensions as **compile-time constants** (`HEIGHT`, `WIDTH`), not runtime parameters. Do **not** include `rows` or `cols` in `function_signature` or `inputs_info`.

Some kernels have additional scalar parameters (e.g. `thresh`, `maxval`, `sigma`, `filter_width`). These are also **not** put in `inputs_info`. They should be:
1. Hardcoded as module-level constants (e.g. `THRESH = 128`)
2. Documented in the `description` string
3. Used directly in the `behavioral()` function

### Step 5 — Write the behavioral() function

#### 5a — Read the C++ source before writing a single line of Python

Open the accel `.cpp` and any header it includes and answer these questions explicitly before writing `behavioral()`:

| Question | What to look for in the source |
|---|---|
| **Division method** | Is the sum divided with integer division (`>>`, `/`, `/ (N*N)`) or converted to float first? Does it truncate or round? |
| **Border normalization** | For pixels near the edge, is the divisor always `FILTER_WIDTH²` (includes zero-padded cells) or is it the count of valid in-bounds pixels only? |
| **Accumulator type** | What C++ type holds the running sum — `uint16_t`, `uint32_t`, `int32_t`? Can it overflow? |
| **Output saturation** | Is the result clamped/saturated explicitly (e.g. `ap_uint<8>` auto-saturates on overflow), or just cast? |
| **Coefficient source** | For weighted filters (Gaussian, Sobel, etc.), are the exact coefficients in the source or a header? Read them — do not use textbook approximations. |

Document each answer as a comment above `behavioral()`. If the source is ambiguous, choose the interpretation that matches a `uint8` truncating cast (the most common HLS pattern) and note it.

#### 5b — Write behavioral() from the source, not from the algorithm name

The `behavioral()` function must be a pure numpy implementation of **what the C++ code actually does**, not what a textbook box/Sobel/Gaussian filter does. It:
- Takes only the **input** buffers as flat numpy arrays (matching the order of input entries in `inputs_info`)
- Returns a flat numpy array of the **output** buffer
- Must NOT depend on any Vitis, HLS, scipy, or cv2 libraries — **numpy only**
- Must reshape inputs using the known `HEIGHT`, `WIDTH` constants before processing
- Must flatten the output before returning
- Must cast the output to the correct numpy dtype using the same truncation/rounding the C++ uses
- Implement all filtering/convolution/etc. manually using `np.pad` and array slicing

### Step 6 — Write the complete generate.py

Use this exact structure:

```python
import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from xf_config_params.h) ---
HEIGHT = <value>
WIDTH = <value>
# Add other relevant constants (FILTER_WIDTH, etc.)

# CRITICAL: Dynarapid ONLY accepts int32_t for all pointer parameters.
# All Vitis types convert to int32_t. Float types (XF_32FC1, ap_fixed, ap_ufixed, float, double) must be SKIPPED.
function_signature = "void <top_function>(int32_t* img_inp, int32_t* img_out)"

description = f"""
This HLS kernel <describe the image processing algorithm in plain terms>.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} <dtype> pixel values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} <dtype> pixel values for
             the output image (same dimensions as input).

<Describe the algorithm precisely enough that a programmer could reimplement it exactly:
  - Border handling: name the strategy (e.g. zero-padding) AND whether border pixels are
    normalized over the full FILTER_WIDTH² window or only over in-bounds pixels.
  - Division/rounding: state whether the result is truncated (floor division / uint cast)
    or rounded, and whether the divisor is a constant or varies per pixel.
  - Coefficients: for weighted filters (Gaussian, Sobel), list the exact integer coefficients
    and any bit-shift used to normalize them.
  - Output type: state how the result is stored (e.g. 'stored as uint8, values above 255
    are saturated' or 'stored as int16, no saturation').
  Do NOT mention Vitis, xf::cv, XF_BORDER_CONSTANT, or any external library name.>
"""

# inputs_info MUST have one entry per pointer parameter in function_signature,
# with the following rules:
#
#   RULE 1 — one input, one output (most common):
#     Two entries: one "in", one "out".
#     behavioral() takes one argument, returns one flat array.
#
#   RULE 2 — multiple inputs, one output (e.g. addweighted: img_in1, img_in2, img_out):
#     Three or more entries: multiple "in", one "out".
#     behavioral() takes one argument per "in" entry, in order.
#
#   RULE 3 — one input, multiple outputs (e.g. sobelfilter: img_inp, img_out1, img_out2):
#     PromptConstructor requires exactly ONE "out" entry.
#     Merge all output buffers into a single combined entry (e.g. shape (2, H, W)).
#     behavioral() returns a single flat concatenated array.
#     The merged entry's "name" is a logical label — it does NOT need to match
#     any C++ parameter name.
#
# Count entries by counting pointer (int32_t*) parameters in function_signature,
# then apply RULE 3 if there are multiple outputs.

inputs_info = [
    # --- Example: one input, one output ---
    {
        "name": "img_inp",            # must match C++ parameter name
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "<numpy_dtype>",
        "iotype": "in",
        "min": 0,
        "max": 255,
    },
    {
        "name": "img_out",            # must match C++ parameter name
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "<numpy_dtype>",
        "iotype": "out",
    },
    # Multiple inputs: add more "in" entries before the "out" entries.
    #   See examples/addweighted/generate.py
    # Multiple outputs: add more "out" entries; behavioral() must return a tuple.
    #   See examples/sobelfilter/generate.py
]

def behavioral(img_inp):              # one argument per "in" entry in inputs_info
    img = img_inp.reshape((HEIGHT, WIDTH))
    # ... numpy implementation of the kernel ...
    result = ...
    return result.astype(np.<numpy_dtype>).flatten()
    # If multiple "out" entries: return a tuple, e.g. return out1.flatten(), out2.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)
pc.write_json("kernel.json")```

---

## Output

Save the generated `generate.py` to:
```
/home/hoodoo17/skills_test/vision_kernels/<top_function_name>/generate.py
```

where `<top_function_name>` is the **C++ function name** extracted from the `function_signature` (e.g., `boxfilter_accel`, not `boxfilter`). This must exactly match the function name in `function_signature` so that the HLS-FAB pipeline can resolve the kernel.

Create the directory if it does not exist.

---

### Step 7 — Verify behavioral() against Dynarapid C simulation

After writing `generate.py`, verify that `behavioral()` produces the expected output for Dynarapid compilation. Full instructions are in `context/csim_verify.md`.

Quick summary:

1. **Create working dir** `/tmp/csim_verify_<top_function_name>/`

2. **Generate test data** — write and run a small Python script that:
   - Creates a random input with `np.random.seed(42)` matching the kernel's input dtype
   - Calls `behavioral()` from the generated `generate.py`
   - Saves input to `input.bin` and expected output to `expected.bin` in the working dir

3. **Write testbench** `tb.cpp` — use the template in `context/csim_verify.md` §3.
   - Forward-declare the kernel function with `int rows, int cols` parameters (the Vitis accel signature always includes these even though `generate.py` does not)
   - Use standard C integer types (`uint8_t`, `int16_t`, `int32_t`, etc.) for pointer declarations
   - Use `HEIGHT`, `WIDTH` from the kernel's `xf_config_params.h`
   - Locate the exact accel `.cpp` filename by listing the kernel's `L1/examples/<kernel_dir>/` directory

4. **Write TCL script** `run_csim.tcl` — use the template in `context/csim_verify.md` §4.
   - Set `KERNEL` to the directory name under `L1/examples/` (e.g. `boxfilter`)
   - Set `TOP` to the C++ function name (e.g. `boxfilter_accel`)
   - Set `ACCEL_CPP` to the exact `.cpp` filename found in step 3

5. **Run csim** (using Vitis HLS, which supports Dynarapid-compatible code):
   ```
   cd /tmp/csim_verify_<top_function_name> && \
   /tools/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls run_csim.tcl 2>&1 | \
   grep -E "PASS|FAIL|mismatch|Max diff|Mismatches|error"
   ```
   This takes 1–3 minutes; the `run_command` timeout is 300 s.

6. **Interpret result**:
   - `PASS` and zero mismatches → log `[VERIFY OK]` and proceed to Step 8.
   - `FAIL` or mismatches > 0 → fix `behavioral()` in `generate.py` and repeat from step 2.
   - Compilation error → check that only standard C types are used (no `ap_uint`, `ap_int`, etc.), check header names, `top_function` spelling, and include paths.

---

## Step 8 — Reconcile the description with the verified behavioral()

Once csim passes, the `behavioral()` code is ground truth. Re-read the `description` string and check it against the verified implementation. Update the description if any of the following are wrong or missing:

1. **Division / rounding** — does the description match what `behavioral()` actually does?
   - If `behavioral()` uses `.mean()` then casts to `uint8` (truncation), the description must say "truncated to uint8" not "rounded".
   - If `behavioral()` uses integer division (`//`), the description must say "integer division (floor)".

2. **Border normalization** — does the description state whether border pixels are divided by the full `FILTER_WIDTH²` or by the count of in-bounds pixels only?
   - Check how the zero-padded window is summed/averaged in `behavioral()` and match that exactly.

3. **Exact coefficients** — for weighted filters (Gaussian, Sobel), does the description list the actual coefficients used in `behavioral()`? If the description says "standard Gaussian" but `behavioral()` uses specific integer weights, replace the vague phrase with the actual numbers.

4. **Output range and saturation** — does the description state the output dtype and whether values are clamped/saturated? Confirm against the `.astype(np.<dtype>)` call in `behavioral()`.

5. **Scalar parameters** — are all hardcoded constants (e.g. `THRESH`, `FILTER_WIDTH`, `SIGMA`) mentioned in the description with their values?

After any description edits, re-write `generate.py` with the updated `description` string (no other changes needed) and log `[DESCRIPTION UPDATED]`, or log `[DESCRIPTION OK]` if no changes were needed.

---

## Important Rules

- **Never** import from Vitis, HLS, scipy, cv2, or any third-party library in `generate.py`. Use only `numpy` and standard Python.
- **Never** mention Vitis Vision, `xf::cv`, `XF_BORDER_CONSTANT`, or any Vitis/HLS type or function in the `description` string. Describe the algorithm in plain terms only (e.g. "zero-padding" not "XF_BORDER_CONSTANT", "box filter" not "xf::cv::boxFilter").
- **CRITICAL — Dynarapid ONLY accepts int32_t.** Use **ONLY `int32_t*` for all pointer parameters** in `function_signature`. Never use `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `uint32_t`, `int64_t`, `uint64_t`, `ap_int`, `ap_uint`, or any floating-point types.
- **CRITICAL — Skip floating-point kernels.** If a kernel uses `float`, `double`, `ap_fixed`, `ap_ufixed`, or any other floating-point type, skip it with reason "Floating-point type not supported by Dynarapid".
- The `dtype` field in `inputs_info` must be `"int32"` (numpy dtype string), not a Vitis or C type.
- `inputs_info` entries must appear in the same order as the pointer (`int32_t*`) parameters in `function_signature` (excluding scalar/runtime params like `rows`, `cols`, `thresh`, `alpha`).
- `inputs_info` has **one entry per pointer** (`int32_t*`) **parameter** in `function_signature`. Entry names must match the C++ parameter names exactly.
- `behavioral()` must accept one argument per `"in"` entry (in order). If there is **one** `"out"` entry, return a single flat array. If there are **multiple** `"out"` entries, return a **tuple** of flat arrays, one per `"out"` entry in order (e.g. `return gx.flatten(), gy.flatten()`).
- Do **not** merge multiple output buffers into a single combined entry — keep them as separate `"out"` entries matching the C++ parameters.
- `shape` should reflect the logical image dimensions, not the C word count.
- Always use the `HEIGHT` and `WIDTH` values from `xf_config_params.h` directly, not hardcoded numerals.
- Do **not** include `#pragma HLS` directives in the generated code; Dynarapid does not recognize them.

---

## Examples

See the `examples/` directory in this skill for complete worked `generate.py` files for:
- `boxfilter` — one input, one output; 2 entries in `inputs_info`; `behavioral()` returns a single flat array
- `addweighted` — two inputs, one output; 3 entries in `inputs_info`; `behavioral()` takes two arguments and returns a single flat array; scalar params (`alpha`, `beta`, `gamma`) hardcoded as constants
- `sobelfilter` — one input, two outputs; 3 entries in `inputs_info`; `behavioral()` returns a **tuple** `(gx.flatten(), gy.flatten())`