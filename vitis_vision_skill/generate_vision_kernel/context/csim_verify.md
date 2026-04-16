# Vitis HLS C-Simulation Verification

After writing `generate.py`, verify that `behavioral()` matches the actual HLS kernel output by running Vitis C simulation (csim). Follow the steps below exactly.

---

## Vitis HLS location

```
/tools/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls
```

---

## Step-by-step process

### 1 — Create a temporary working directory

```
/tmp/csim_verify_<kernel_name>/
```

### 2 — Generate test input and expected output using the generate.py behavioral()

Write a Python script `/tmp/csim_verify_<kernel_name>/gen_test_data.py` with this content (adapting imports from the kernel's `inputs_info`):

```python
import sys, importlib.util, numpy as np
sys.path.insert(0, "/home/plwalker/skills_test/vision_kernels/<top_function_name>")

# Load generate.py to get behavioral(), HEIGHT, WIDTH, inputs_info
spec = importlib.util.spec_from_file_location(
    "gen", "/home/plwalker/skills_test/vision_kernels/<top_function_name>/generate.py")
gen = importlib.util.load_from_spec(spec)
spec.loader.exec_module(gen)

np.random.seed(42)
inp = np.random.randint(
    gen.inputs_info[0]["min"], gen.inputs_info[0]["max"] + 1,
    gen.inputs_info[0]["total_elements"], dtype=np.uint8)

inp.tofile("/tmp/csim_verify_<top_function_name>/input.bin")

expected = gen.behavioral(inp)
expected.tofile("/tmp/csim_verify_<top_function_name>/expected.bin")

print(f"Input:    {inp[:8].tolist()}")
print(f"Expected: {expected[:8].tolist()}")
```

Run it with: `/home/plwalker/HLS-FAB/venv/bin/python3 /tmp/csim_verify_<top_function_name>/gen_test_data.py`

### 3 — Write the C++ testbench

Write `/tmp/csim_verify_<top_function_name>/tb.cpp`:

```cpp
#include <cstdio>
#include <cstdint>
#include "xf_<kernel>_accel_config.h"   // adjust filename to match actual config header

// Forward-declare the HLS top function exactly as it appears in the accel .cpp
void <top_function>(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                    int rows, int cols);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    uint8_t inp_raw[NPIX];
    FILE* f = fopen(argv[1], "rb");
    fread(inp_raw, 1, NPIX, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH>  img_inp[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NPIX];
    for (int i = 0; i < NPIX; i++) img_inp[i] = inp_raw[i];

    <top_function>(img_inp, img_out, HEIGHT, WIDTH);

    uint8_t out_raw[NPIX];
    for (int i = 0; i < NPIX; i++) out_raw[i] = (uint8_t)(img_out[i].to_uint());

    f = fopen(argv[2], "wb"); fwrite(out_raw, 1, NPIX, f); fclose(f);

    uint8_t exp_raw[NPIX];
    f = fopen(argv[3], "rb"); fread(exp_raw, 1, NPIX, f); fclose(f);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX; i++) {
        int d = abs((int)out_raw[i] - (int)exp_raw[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1) {
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n",
                       i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>1): %d / %d\n", mismatches, NPIX);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
```

**Important**: the `<top_function>` forward-declaration must include `int rows, int cols` because that is how the Vitis accel source declares it, even though `generate.py` omits those from `function_signature`.

### 4 — Write the Vitis HLS TCL script

Write `/tmp/csim_verify_<top_function_name>/run_csim.tcl`:

```tcl
set XF_ROOT /home/plwalker/Vitis_Libraries/vision
set KERNEL  <kernel_dir>          ;# directory name under L1/examples/
set TOP     <top_function>        ;# C++ function name
set ACCEL_CPP "${XF_ROOT}/L1/examples/${KERNEL}/xf_<kernel>_accel.cpp"
set CFG_DIR   "${XF_ROOT}/L1/examples/${KERNEL}/config"
set INC_DIR   "${XF_ROOT}/L1/examples/${KERNEL}"
set TB_CPP    "/tmp/csim_verify_${TOP}/tb.cpp"

set CFLAGS "-I ${CFG_DIR} -I ${INC_DIR} -I ${XF_ROOT}/L1/include -D__SDSVHLS__ -std=c++14"

open_project -reset "csim_${TOP}.prj"
add_files      "${ACCEL_CPP}" -cflags "${CFLAGS}" -csimflags "${CFLAGS}"
add_files -tb  "${TB_CPP}"    -cflags "${CFLAGS}" -csimflags "${CFLAGS}"
set_top ${TOP}
open_solution -reset sol1
set_part xcu200-fsgd2104-2-e
create_clock -period 3.3

csim_design -argv "/tmp/csim_verify_${TOP}/input.bin /tmp/csim_verify_${TOP}/output.bin /tmp/csim_verify_${TOP}/expected.bin"
exit
```

### 5 — Run the simulation

```bash
cd /tmp/csim_verify_<top_function_name> && \
/tools/Xilinx/Vitis_HLS/2024.2/bin/vitis_hls run_csim.tcl 2>&1 | \
grep -E "PASS|FAIL|mismatch|Max diff|Mismatches|SIM 211|error"
```

### 6 — Interpret results

- `PASS` + zero mismatches → `behavioral()` is correct. Log `[VERIFY OK]`.
- `FAIL` or mismatches > 0 → `behavioral()` is wrong. Fix it and re-run verification.
- Compilation error → check that the config header name and `top_function` forward-declaration are correct.

---

## Notes

- The testbench always passes `rows=HEIGHT, cols=WIDTH` to the kernel — those are compile-time constants in `xf_config_params.h`.
- The ±1 tolerance in the mismatch check accounts for float rounding differences between C and Python; the Vitis testbench itself uses the same tolerance.
- csim typically takes 1–3 minutes for a 128×128 image.
- The accel `.cpp` filename pattern is `xf_<something>_accel.cpp` — read the directory listing to get the exact name.
