#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
// The accel .cpp includes xf_houghlines_accel_config.h which includes xf_config_params.h
// Use the config header directly
#include "xf_config_params.h"

// Forward-declare the HLS top function exactly as it appears in the accel .cpp
void houghlines_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                      short threshold,
                      short maxlines,
                      float* arrayy,
                      float* arrayx,
                      int rows,
                      int cols);

static constexpr int NPIX = HEIGHT * WIDTH;
static constexpr int NLINES = LINESMAX;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    // Read input image
    uint8_t inp_raw[NPIX];
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fread(inp_raw, 1, NPIX, f);
    fclose(f);

    ap_uint<INPUT_PTR_WIDTH> img_inp[NPIX];
    for (int i = 0; i < NPIX; i++) img_inp[i] = inp_raw[i];

    float arrayy[NLINES];
    float arrayx[NLINES];
    memset(arrayy, 0, sizeof(arrayy));
    memset(arrayx, 0, sizeof(arrayx));

    // Run kernel
    houghlines_accel(img_inp, (short)20, (short)512, arrayy, arrayx, HEIGHT, WIDTH);

    // Write output: arrayy then arrayx concatenated as float32
    f = fopen(argv[2], "wb");
    fwrite(arrayy, sizeof(float), NLINES, f);
    fwrite(arrayx, sizeof(float), NLINES, f);
    fclose(f);

    // Read expected
    float exp_arrayy[NLINES];
    float exp_arrayx[NLINES];
    f = fopen(argv[3], "rb");
    fread(exp_arrayy, sizeof(float), NLINES, f);
    fread(exp_arrayx, sizeof(float), NLINES, f);
    fclose(f);

    // Compare
    int mismatches = 0;
    float max_diff_rho = 0.0f, max_diff_theta = 0.0f;
    for (int i = 0; i < NLINES; i++) {
        float dr = fabsf(arrayy[i] - exp_arrayy[i]);
        float dt = fabsf(arrayx[i] - exp_arrayx[i]);
        if (dr > max_diff_rho) max_diff_rho = dr;
        if (dt > max_diff_theta) max_diff_theta = dt;
        // Tolerance: rho within 2.0, theta within 0.02 rad
        if (dr > 2.0f || dt > 0.02f) {
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch[%d]: kernel rho=%f theta=%f | expected rho=%f theta=%f\n",
                       i, arrayy[i], arrayx[i], exp_arrayy[i], exp_arrayx[i]);
        }
    }
    printf("Max rho diff: %f\n", max_diff_rho);
    printf("Max theta diff: %f\n", max_diff_theta);
    printf("Mismatches: %d / %d\n", mismatches, NLINES);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
