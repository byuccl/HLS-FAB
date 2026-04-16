# Vitis HLS C-Simulation Verification (DynaRapid int32 Version)

After writing `generate.py`, verify that `behavioral()` produces the same output as
the original Vitis kernel (accounting for int32 container conversion). DynaRapid does
not have its own simulator, so we verify against the Vitis HLS C simulation.

The key difference from Vitis verification: the testbench must convert between the
original Vitis types (uint8, int16, etc.) and int32 for comparison with `behavioral()`.

---

## Vitis HLS location

```
${VITIS_HLS}/bin/vitis_hls
```

---

## Step-by-step process

### 1 — Create a temporary working directory

```
/tmp/csim_verify_<kernel_name>_int32/
```

### 2 — Generate test data using the generate.py behavioral()

Write a Python script `/tmp/csim_verify_<kernel_name>_int32/gen_test_data.py`:

```python
import sys, importlib.util, numpy as np

# Load generate.py
spec = importlib.util.spec_from_file_location(
    "gen", "${REPO_ROOT}/dataset/kernels/<top_function_name>_int32/generate.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)

np.random.seed(42)

# Generate input in the LOGICAL pixel range (e.g. 0-255), stored as int32
inp = np.random.randint(
    gen.inputs_info[0]["min"], gen.inputs_info[0]["max"] + 1,
    gen.inputs_info[0]["total_elements"], dtype=np.int32)

inp.tofile("/tmp/csim_verify_<top_function_name>_int32/input.bin")

expected = gen.behavioral(inp)
if isinstance(expected, tuple):
    for i, e in enumerate(expected):
        e.tofile(f"/tmp/csim_verify_<top_function_name>_int32/expected_{i}.bin")
else:
    expected.tofile("/tmp/csim_verify_<top_function_name>_int32/expected.bin")

print(f"Input:    {inp[:8].tolist()}")
if isinstance(expected, tuple):
    for i, e in enumerate(expected):
        print(f"Expected[{i}]: {e[:8].tolist()}")
else:
    print(f"Expected: {expected[:8].tolist()}")
```

Run it with: `${REPO_ROOT}/venv/bin/python3 /tmp/csim_verify_<top_function_name>_int32/gen_test_data.py`

### 3 — Write the C++ testbench

Write `/tmp/csim_verify_<top_function_name>_int32/tb.cpp`.

The testbench must:
1. Read int32 input from `input.bin`
2. Convert int32 to the original Vitis type (e.g. cast int32 → uint8) for the kernel call
3. Call the original Vitis kernel (which uses `ap_uint<N>*`)
4. Convert the kernel output back to int32
5. Compare against `expected.bin` from `behavioral()`

```cpp
#include <cstdio>
#include <cstdint>
#include "xf_<kernel>_accel_config.h"

// Forward-declare the HLS top function (original Vitis signature)
void <top_function>(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                    int rows, int cols);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    // Read int32 input (from behavioral() test data)
    int32_t inp_int32[NPIX];
    FILE* f = fopen(argv[1], "rb");
    fread(inp_int32, sizeof(int32_t), NPIX, f); fclose(f);

    // Convert int32 → original Vitis type for kernel call
    ap_uint<INPUT_PTR_WIDTH>  img_inp[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NPIX];
    for (int i = 0; i < NPIX; i++) img_inp[i] = (uint8_t)inp_int32[i];

    <top_function>(img_inp, img_out, HEIGHT, WIDTH);

    // Convert kernel output → int32 for comparison
    int32_t out_int32[NPIX];
    for (int i = 0; i < NPIX; i++) out_int32[i] = (int32_t)(uint8_t)(img_out[i].to_uint());

    f = fopen(argv[2], "wb"); fwrite(out_int32, sizeof(int32_t), NPIX, f); fclose(f);

    // Read expected int32 output from behavioral()
    int32_t exp_int32[NPIX];
    f = fopen(argv[3], "rb"); fread(exp_int32, sizeof(int32_t), NPIX, f); fclose(f);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX; i++) {
        int d = abs(out_int32[i] - exp_int32[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1) {
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n",
                       i, out_int32[i], exp_int32[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>1): %d / %d\n", mismatches, NPIX);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
```

**Important**: Adapt the cast types based on the kernel:
- For `XF_8UC1` input: cast `(uint8_t)inp_int32[i]`
- For `XF_16SC1` output: cast `(int32_t)(int16_t)(img_out[i].to_uint())`
- For `XF_16UC1`: cast `(int32_t)(uint16_t)(img_out[i].to_uint())`

### 4 — Write the Vitis HLS TCL script

Write `/tmp/csim_verify_<top_function_name>_int32/run_csim.tcl`:

```tcl
set XF_ROOT ${VITIS_LIBRARIES}/vision
set KERNEL  <kernel_dir>
set TOP     <top_function>
set ACCEL_CPP "${XF_ROOT}/L1/examples/${KERNEL}/xf_<kernel>_accel.cpp"
set CFG_DIR   "${XF_ROOT}/L1/examples/${KERNEL}/config"
set INC_DIR   "${XF_ROOT}/L1/examples/${KERNEL}"
set TB_CPP    "/tmp/csim_verify_${TOP}_int32/tb.cpp"

set CFLAGS "-I ${CFG_DIR} -I ${INC_DIR} -I ${XF_ROOT}/L1/include -D__SDSVHLS__ -std=c++14"

open_project -reset "csim_${TOP}_int32.prj"
add_files      "${ACCEL_CPP}" -cflags "${CFLAGS}" -csimflags "${CFLAGS}"
add_files -tb  "${TB_CPP}"    -cflags "${CFLAGS}" -csimflags "${CFLAGS}"
set_top ${TOP}
open_solution -reset sol1
set_part xcu200-fsgd2104-2-e
create_clock -period 3.3

csim_design -argv "/tmp/csim_verify_${TOP}_int32/input.bin /tmp/csim_verify_${TOP}_int32/output.bin /tmp/csim_verify_${TOP}_int32/expected.bin"
exit
```

### 5 — Run the simulation

```bash
cd /tmp/csim_verify_<top_function_name>_int32 && \
${VITIS_HLS}/bin/vitis_hls run_csim.tcl 2>&1 | \
grep -E "PASS|FAIL|mismatch|Max diff|Mismatches|SIM 211|error"
```

### 6 — Interpret results

- `PASS` + zero mismatches → `behavioral()` is correct. Log `[VERIFY OK]`.
- `FAIL` or mismatches > 0 → `behavioral()` is wrong. Fix it and re-run.
- Compilation error → check header names, `top_function` spelling, include paths.

---

## Notes

- The testbench passes `rows=HEIGHT, cols=WIDTH` to the Vitis kernel.
- The ±1 tolerance accounts for float rounding differences between C and Python.
- csim typically takes 1–3 minutes for a 128×128 image.
- The accel `.cpp` filename pattern is `xf_<something>_accel.cpp` — list the directory to get the exact name.
