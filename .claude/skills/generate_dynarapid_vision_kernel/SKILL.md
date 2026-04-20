---
name: generate_dynarapid_vision_kernel
description: Generates a generate.py file for one, a listed subset, or all Vitis Vision Library L1 kernels adapted for DynaRapid HLS (int32-only), compatible with the HLS-FAB dataset format.
---
# Generate DynaRapid Vision Kernel Skill

Use this skill to generate a `generate.py` file for one or more kernels from the Xilinx Vitis Vision Library, **adapted for DynaRapid HLS compilation**.

DynaRapid only supports `int32_t` for all pointer parameters. This skill converts the original Vitis Vision kernel types (typically `uint8`, `int16`, etc.) to `int32_t` while **preserving the original image-processing logic** — pixel values that would normally be 8-bit (0–255) or 16-bit are stored inside `int32_t` containers, and the behavioral model clamps/masks them to their logical range.

Each `generate.py` is part of the HLS-FAB dataset and describes a kernel's function signature, I/O buffers, behavioral Python model, and example data for AI-assisted HLS code generation.

---

## Path Resolution (MUST run before any other step)

Before doing anything else, resolve the two required external paths. Check **environment variables first**, then fall back to the **config file**.

1. Check if the environment variables `VITIS_LIBRARIES` and `VITIS_HLS` are set:
   ```bash
   echo "VITIS_LIBRARIES=${VITIS_LIBRARIES:-NOT SET}"
   echo "VITIS_HLS=${VITIS_HLS:-NOT SET}"
   ```

2. For any variable that is **NOT SET**, read `vision_skill_paths.cfg` in the repository root and parse the values from there. The config file format is `KEY=VALUE` (lines starting with `#` are comments).

3. If a path is neither in the environment nor in the config file, **stop and ask the user** to either:
   - Set the environment variable (`export VITIS_LIBRARIES=/path/to/Vitis_Libraries`), or
   - Edit `vision_skill_paths.cfg` in the repository root (copy from `vision_skill_paths.cfg.example` if it doesn't exist).

4. **Validate** that the resolved paths exist before proceeding. If they don't, tell the user which path is invalid.

Use these resolved values everywhere this skill references `${VITIS_LIBRARIES}` or `${VITIS_HLS}`.

All other paths are relative to the HLS-FAB project root (`${REPO_ROOT}`, i.e. the current working directory):

- **Kernel source**: `${VITIS_LIBRARIES}/vision/L1/examples/<kernel>/`
- **Output**: `${REPO_ROOT}/dataset/kernels/<top_function_name>_int32/generate.py`
- **Python venv**: `${REPO_ROOT}/venv/bin/python3`

---

## Input

The user will provide one of:

- A **single kernel name** — a directory name under `${VITIS_LIBRARIES}/vision/L1/examples/`. For example:
  - `boxfilter`
  - `sobelfilter`
  - `gaussianfilter`
  - `resize`
  - `threshold`

- The keyword **`all`** — to generate `generate.py` for every kernel directory found under `${VITIS_LIBRARIES}/vision/L1/examples/`, skipping any kernel whose output file already exists at `dataset/kernels/<top_function_name>_int32/generate.py`.

- A path to a **text file** (e.g. `@kernels_todo.txt`) — to generate `generate.py` for each kernel listed in the file, one kernel directory name per line. Lines starting with `#` and blank lines are ignored. Kernels whose output file already exists are skipped with `[SKIP]`.

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

1. List every subdirectory of `${VITIS_LIBRARIES}/vision/L1/examples/` (excluding `readme` and any non-kernel directories) to build the kernel list.

When the input is `@<filepath>`:

1. Read the file at `<filepath>`. Split on newlines; strip each line; discard lines that are blank or start with `#`. The remaining lines are the kernel list.

For both modes, then for each kernel in the list:

   a. Read the kernel's `description.json` (or accel `.cpp`) to determine the **top function name**. Check if `dataset/kernels/<top_function_name>_int32/generate.py` already exists. If it does, skip it and log `[SKIP] <top_function_name>_int32`.
   b. Otherwise, run the full single-kernel process (Steps 1–8 below) for that kernel.
   c. Log `[OK] <kernel_name>` on success, `[SKIP-FLOAT] <kernel_name>` if skipped due to floating-point types, or `[FAIL] <kernel_name>: <reason>` on failure — then continue to the next kernel without stopping.

After all kernels are processed, print a summary: total processed, skipped, skipped-float, succeeded, and failed.

---

## Process

> **CRITICAL — read before writing any file:**
> - **DynaRapid ONLY accepts `int32_t` types.** All pointer parameters in `function_signature` must be `int32_t*`. All `dtype` fields in `inputs_info` must be `"int32"`.
> - **Skip any kernel using floating-point types** (`float`, `double`, `ap_fixed`, `ap_ufixed`, `XF_32FC1`).
> - The `function_signature` must contain **only** the buffer pointer parameters (`int32_t*`). **Never include `rows`, `cols`, or any other runtime dimension parameter** — image dimensions are fixed compile-time constants (`HEIGHT`, `WIDTH`).
> - The `description` must **not** mention `rows` or `cols` as parameters.
> - The Vitis Vision C++ source uses `rows`/`cols` as runtime args, but the HLS-FAB format does not. Ignore them.

Follow these steps for every kernel:

### Step 0 — Check for existing Vitis generate.py (shortcut)

Before reading C++ source from scratch, check if a Vitis HLS `generate.py` already exists:

```
${REPO_ROOT}/dataset/all_vision_kernels/<top_function_name>_accel/generate.py
```

(or similar name variants — check both `<kernel>_accel` and `<top_function>` patterns.)

If found, **use it as the starting point**: the `behavioral()`, `description`, `inputs_info`, and algorithm details are already implemented. The DynaRapid adaptation is then mechanical:
1. Change all `dtype` fields from `"uint8"` / `"int16"` etc. to `"int32"`
2. Change pointer types in `function_signature` from `ap_uint<N>*` to `int32_t*`
3. Add `_int32` suffix to the function name
4. Change numpy casts from `.astype(np.uint8)` to `.astype(np.int32)`
5. Adjust `min`/`max` ranges if needed (keep the logical pixel range, e.g. 0–255)
6. Optionally reduce `HEIGHT`/`WIDTH` for DynaRapid-friendly sizes (e.g. 128x128)

This is much faster and more reliable than reverse-engineering from C++ source.

### Step 1 — Read the kernel source files

For a kernel named `<kernel>`, read all of the following files (they may not all exist, use what is available):
- `${VITIS_LIBRARIES}/vision/L1/examples/<kernel>/xf_*_accel.cpp` — the top-level HLS kernel implementation
- `${VITIS_LIBRARIES}/vision/L1/examples/<kernel>/xf_*_accel_config.h` — main config header (includes `xf_config_params.h`)
- `${VITIS_LIBRARIES}/vision/L1/examples/<kernel>/config/xf_config_params.h` — defines `HEIGHT`, `WIDTH`, `IN_TYPE`, `OUT_TYPE`, `INPUT_PTR_WIDTH`, `OUTPUT_PTR_WIDTH`, `NPPCX`, `FILTER_WIDTH`, etc.
- `${VITIS_LIBRARIES}/vision/L1/examples/<kernel>/description.json` — contains `topfunction` and other metadata

### Step 2 — Extract key parameters

From the source files, identify:

| Parameter | Where to find it |
|---|---|
| Top function name | `description.json` → `topfunction`, or the `void <name>(...)` in the accel `.cpp` |
| `HEIGHT`, `WIDTH` | `xf_config_params.h` |
| `IN_TYPE`, `OUT_TYPE` | `xf_config_params.h` (e.g. `XF_8UC1`, `XF_16SC1`) |
| `INPUT_PTR_WIDTH`, `OUTPUT_PTR_WIDTH` | `xf_config_params.h` — read for reference only; **do NOT use in the function signature** |
| `NPPCX` | `xf_config_params.h` |
| Scalar parameters | Additional `int`, `unsigned char`, `float`, etc. arguments in the function signature (e.g. `thresh`, `maxval`, `sigma`) |
| Filter width | `FILTER_WIDTH` or similar defines |

Also determine the **original logical pixel range** from `IN_TYPE`/`OUT_TYPE`:
- `XF_8UC1` / `XF_8UC3` → logical range 0–255 (8-bit unsigned)
- `XF_16UC1` → logical range 0–65535 (16-bit unsigned)
- `XF_16SC1` → logical range -32768–32767 (16-bit signed)
- `XF_32SC1` → logical range -2³¹–2³¹-1 (32-bit signed, native)

### Step 3 — Skip or convert: determine DynaRapid compatibility

**Skip the kernel entirely** if it uses any of:
- `XF_32FC1`, `XF_64FC1`, `float`, `double`, `ap_fixed`, `ap_ufixed`
- Log `[SKIP-FLOAT] <kernel_name>: Floating-point type not supported by DynaRapid`

**Convert all other types to `int32_t`:**

| Vitis Type | DynaRapid Type | numpy dtype | Logical Range | Channels |
|---|---|---|---|---|
| `XF_8UC1` | `int32_t` | `int32` | 0–255 | 1 |
| `XF_8UC3` | `int32_t` | `int32` | 0–255 | 3 |
| `XF_16UC1` | `int32_t` | `int32` | 0–65535 | 1 |
| `XF_16SC1` | `int32_t` | `int32` | -32768–32767 | 1 |
| `XF_32SC1` | `int32_t` | `int32` | -2³¹–2³¹-1 | 1 |

> **CRITICAL — `int32_t` ONLY.**
>
> - Do NOT use `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `uint32_t`, `int64_t`, `uint64_t`.
> - Do NOT use `ap_int<N>`, `ap_uint<N>`, `ap_fixed`, or `ap_ufixed`.
> - Do NOT use any floating-point types (`float`, `double`).
> - Always use `int32_t*` for every pointer parameter.

### Step 4 — Identify scalar (non-buffer) parameters

Unlike the Vitis Vision C++ source, the HLS-FAB `generate.py` treats image dimensions as **compile-time constants** (`HEIGHT`, `WIDTH`), not runtime parameters. Do **not** include `rows` or `cols` in `function_signature` or `inputs_info`.

Some kernels have additional scalar parameters (e.g. `thresh`, `maxval`, `sigma`, `filter_width`). These are also **not** put in `inputs_info`. They should be:
1. Hardcoded as module-level constants (e.g. `THRESH = 128`)
2. Documented in the `description` string
3. Used directly in the `behavioral()` function

### Step 5 — Write the behavioral() function

#### 5a — Determine the original logical pixel range

Before writing any code, record the **logical pixel range** from Step 2. This is critical because int32 is just a container — the algorithm must still operate on values in the original range.

For example, a box filter on a `XF_8UC1` image:
- Input int32 values represent pixels in range [0, 255]
- The mean filter operates on these 0–255 values
- The output must be clamped to [0, 255] before storing as int32

Document the logical range as a comment:
```python
# Original Vitis type: XF_8UC1 → logical pixel range [0, 255]
# Values are stored in int32 but represent uint8 pixels.
# All intermediate computations preserve this semantic.
PIXEL_MIN = 0
PIXEL_MAX = 255
```

For kernels where input and output types differ (e.g. Sobel: input `XF_8UC1` → output `XF_16SC1`):
```python
# Input: XF_8UC1 → logical range [0, 255]
# Output: XF_16SC1 → logical range [-32768, 32767]
INPUT_PIXEL_MIN = 0
INPUT_PIXEL_MAX = 255
OUTPUT_PIXEL_MIN = -32768
OUTPUT_PIXEL_MAX = 32767
```

#### 5b — Read the C++ source before writing a single line of Python

Open the accel `.cpp` and any header it includes and answer these questions explicitly before writing `behavioral()`:

| Question | What to look for in the source |
|---|---|
| **Division method** | Is the sum divided with integer division (`>>`, `/`, `/ (N*N)`) or converted to float first? Does it truncate or round? |
| **Border normalization** | For pixels near the edge, is the divisor always `FILTER_WIDTH²` (includes zero-padded cells) or is it the count of valid in-bounds pixels only? |
| **Accumulator type** | What C++ type holds the running sum — `uint16_t`, `uint32_t`, `int32_t`? Can it overflow? |
| **Output saturation** | Is the result clamped/saturated explicitly (e.g. `ap_uint<8>` auto-saturates on overflow), or just cast? |
| **Coefficient source** | For weighted filters (Gaussian, Sobel, etc.), are the exact coefficients in the source or a header? Read them — do not use textbook approximations. |

Document each answer as a comment above `behavioral()`. If the source is ambiguous, choose the interpretation that matches the original type's truncating cast (the most common HLS pattern) and note it.

#### 5c — Write behavioral() preserving original logic in int32

The `behavioral()` function must be a pure numpy implementation of **what the C++ code actually does**, adapted for int32 containers:

- Takes only the **input** buffers as flat `int32` numpy arrays
- Returns a flat `int32` numpy array of the **output** buffer
- Must NOT depend on any Vitis, HLS, scipy, or cv2 libraries — **numpy only**
- Must reshape inputs using the known `HEIGHT`, `WIDTH` constants before processing
- Must flatten the output before returning
- **Must clamp the output to the logical pixel range** before returning (e.g. `np.clip(result, 0, 255).astype(np.int32)` for uint8-origin data)
- Implement all filtering/convolution/etc. manually using `np.pad` and array slicing

**Key pattern — preserving sub-32-bit logic in int32:**

```python
def behavioral(img_inp):
    # Reshape the int32 input (values logically in [0, 255])
    img = img_inp.reshape((HEIGHT, WIDTH))
    
    # Process using the same arithmetic as the original kernel
    # (intermediate computations may exceed [0, 255] temporarily)
    pad = FILTER_WIDTH // 2
    padded = np.pad(img, pad, mode="constant", constant_values=0)
    out = np.zeros((HEIGHT, WIDTH), dtype=np.int64)  # use int64 accumulator to avoid overflow
    for i in range(HEIGHT):
        for j in range(WIDTH):
            window_sum = padded[i:i+FILTER_WIDTH, j:j+FILTER_WIDTH].sum()
            out[i, j] = window_sum // (FILTER_WIDTH * FILTER_WIDTH)  # integer division
    
    # Clamp to the original logical range, then store as int32
    return np.clip(out, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()
```

### Step 6 — Write the complete generate.py

Use this exact structure:

```python
import numpy as np
from hlsfab import PromptConstructor, generate_examples

# --- Constants (from xf_config_params.h) ---
HEIGHT = <value>
WIDTH = <value>
# Add other relevant constants (FILTER_WIDTH, etc.)

# --- Logical pixel range ---
# Original Vitis type: <XF_TYPE> → logical range [MIN, MAX]
# Values are stored in int32 but represent <original_type> pixels.
PIXEL_MIN = <min>
PIXEL_MAX = <max>

# CRITICAL: DynaRapid ONLY accepts int32_t for all pointer parameters.
function_signature = "void <top_function>_int32(int32_t* img_inp, int32_t* img_out)"

description = f"""
This HLS kernel <describe the image processing algorithm in plain terms>.

Parameters:
    img_inp: Pointer to a flat array of {HEIGHT * WIDTH} int32 values representing
             a {HEIGHT}x{WIDTH} grayscale input image in row-major order.
             Each value represents a pixel in the range [{PIXEL_MIN}, {PIXEL_MAX}].
    img_out: Pointer to a flat array of {HEIGHT * WIDTH} int32 values for
             the output image (same dimensions as input).
             Each output value is clamped to [{PIXEL_MIN}, {PIXEL_MAX}].

<Describe the algorithm precisely:
  - State that pixel values are logically <original_bit_width>-bit stored in int32.
  - Border handling, division/rounding, coefficients, output saturation — same
    precision requirements as the Vitis version.
  Do NOT mention Vitis, xf::cv, XF_BORDER_CONSTANT, or any external library name.>
"""

inputs_info = [
    {
        "name": "img_inp",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "in",
        "min": PIXEL_MIN,
        "max": PIXEL_MAX,
    },
    {
        "name": "img_out",
        "array_dimension": 2,
        "shape": (HEIGHT, WIDTH),
        "total_elements": HEIGHT * WIDTH,
        "dtype": "int32",
        "iotype": "out",
    },
]

def behavioral(img_inp):
    img = img_inp.reshape((HEIGHT, WIDTH))
    # ... numpy implementation preserving original logic ...
    result = ...
    return np.clip(result, PIXEL_MIN, PIXEL_MAX).astype(np.int32).flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)
pc.write_json("kernel.json")
```

### Naming Convention

The top function name and output directory must have an `_int32` suffix to distinguish from Vitis versions:
- Vitis: `boxfilter_accel` → DynaRapid: `boxfilter_accel_int32`
- Output path: `dataset/kernels/boxfilter_accel_int32/generate.py`

---

## Output

Save the generated `generate.py` to:
```
dataset/kernels/<top_function_name>_int32/generate.py
```

Create the directory if it does not exist.

---

## Step 7 — Verify behavioral() against Vitis C simulation

After writing `generate.py`, run Vitis HLS C simulation to confirm that `behavioral()` produces the same output as the compiled kernel (accounting for int32 container conversion). Full instructions are in `context/csim_verify.md`.

Quick summary:

1. **Create working dir** `/tmp/csim_verify_<top_function_name>_int32/`

2. **Generate test data** — write and run a small Python script that:
   - Creates random input with `np.random.seed(42)` in the logical pixel range
   - Calls `behavioral()` from the generated `generate.py`
   - Saves input to `input.bin` and expected output to `expected.bin` in the working dir

3. **Write testbench** `tb.cpp` — use the template in `context/csim_verify.md` §3.
   - The testbench must convert between the original Vitis types and int32 for comparison
   - Use `HEIGHT`, `WIDTH` from the kernel's `xf_config_params.h`

4. **Write TCL script** `run_csim.tcl` — use the template in `context/csim_verify.md` §4.

5. **Run csim**:
   ```
   cd /tmp/csim_verify_<top_function_name>_int32 && \
   ${VITIS_HLS}/bin/vitis_hls run_csim.tcl 2>&1 | \
   grep -E "PASS|FAIL|mismatch|Max diff|Mismatches|error"
   ```

6. **Interpret result**:
   - `PASS` and zero mismatches → log `[VERIFY OK]` and proceed to Step 8.
   - `FAIL` or mismatches > 0 → fix `behavioral()` in `generate.py` and repeat from step 2.

---

## Step 8 — Reconcile the description with the verified behavioral()

Once csim passes, the `behavioral()` code is ground truth. Re-read the `description` string and check it against the verified implementation. Update the description if any of the following are wrong or missing:

1. **Division / rounding** — does the description match what `behavioral()` actually does?
2. **Border normalization** — does the description state whether border pixels are divided by the full `FILTER_WIDTH²` or by the count of in-bounds pixels only?
3. **Exact coefficients** — for weighted filters, does the description list the actual coefficients?
4. **Output range and saturation** — does the description state the output range and clamping?
5. **Scalar parameters** — are all hardcoded constants mentioned with their values?
6. **Logical pixel range** — does the description clearly state that int32 values represent pixels in the original range?

After any description edits, re-write `generate.py` with the updated `description` string and log `[DESCRIPTION UPDATED]`, or log `[DESCRIPTION OK]` if no changes were needed.

---

## Important Rules

- **Never** import from Vitis, HLS, scipy, cv2, or any third-party library in `generate.py`. Use only `numpy` and standard Python.
- **Never** mention Vitis Vision, `xf::cv`, `XF_BORDER_CONSTANT`, or any Vitis/HLS type or function in the `description` string. Describe the algorithm in plain terms only.
- **CRITICAL — DynaRapid ONLY accepts `int32_t`.** Use **ONLY `int32_t*` for all pointer parameters** in `function_signature`. Never use `int8_t`, `uint8_t`, `int16_t`, `uint16_t`, `uint32_t`, `int64_t`, `uint64_t`, `ap_int`, `ap_uint`, or any floating-point types.
- **CRITICAL — Skip floating-point kernels.** If a kernel uses `float`, `double`, `ap_fixed`, `ap_ufixed`, or any other floating-point type, skip it.
- **CRITICAL — Preserve logical pixel range.** Input `min`/`max` in `inputs_info` must reflect the original pixel range (e.g. 0–255 for `XF_8UC1`), NOT the full int32 range. The `behavioral()` function must clamp outputs to the logical range.
- The `dtype` field in `inputs_info` must always be `"int32"`.
- `inputs_info` entries must appear in the same order as the pointer (`int32_t*`) parameters in `function_signature`.
- `inputs_info` has **one entry per pointer (`int32_t*`) parameter** in `function_signature`. Entry names must match the C++ parameter names exactly.
- `behavioral()` must accept one argument per `"in"` entry (in order). If there is **one** `"out"` entry, return a single flat array. If there are **multiple** `"out"` entries, return a **tuple** of flat arrays.
- `shape` should reflect the logical image dimensions, not the C word count.
- Always use the `HEIGHT` and `WIDTH` values from `xf_config_params.h` directly, not hardcoded numerals.
- Do **not** include `#pragma HLS` directives — DynaRapid does not recognize them.
- The function name must have an `_int32` suffix (e.g. `boxfilter_accel_int32`).

---

## Examples

See the `examples/` directory in this skill for complete worked `generate.py` files for:
- `boxfilter` — one input, one output; 2 entries in `inputs_info`; `behavioral()` returns a single flat array; input pixels logically in [0, 255]
- `addweighted` — two inputs, one output; 3 entries in `inputs_info`; `behavioral()` takes two arguments; scalar params hardcoded as constants; pixels logically in [0, 255]
- `sobelfilter` — one input, two outputs; 3 entries in `inputs_info`; `behavioral()` returns a **tuple**; input pixels [0, 255], output gradients [-32768, 32767]
