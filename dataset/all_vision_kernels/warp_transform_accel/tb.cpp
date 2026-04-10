#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "xf_warp_transform_accel_config.h"

// Forward-declare the HLS top function exactly as in xf_warp_transform_accel.cpp
void warp_transform_accel(
    ap_uint<INPUT_PTR_WIDTH>* img_in,
    float* transform,
    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
    int rows, int cols);

static constexpr int NPIX = HEIGHT * WIDTH;
// INPUT_PTR_WIDTH=32, 4 bytes per word => NPIX/4 words for input/output
static constexpr int NWORDS = NPIX / (INPUT_PTR_WIDTH / 8);

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <input.bin> <transform.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    // Read raw pixel input
    uint8_t inp_raw[NPIX];
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fread(inp_raw, 1, NPIX, f); fclose(f);

    // Read transform matrix (9 floats)
    float transform[9];
    f = fopen(argv[2], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[2]); return 1; }
    fread(transform, sizeof(float), 9, f); fclose(f);

    // Pack input pixels into ap_uint<32> words (4 bytes per word)
    ap_uint<INPUT_PTR_WIDTH>  img_in[NWORDS];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NWORDS];
    for (int i = 0; i < NWORDS; i++) {
        img_in[i] = 0;
        for (int b = 0; b < 4; b++) {
            img_in[i].range(b*8+7, b*8) = inp_raw[i*4 + b];
        }
        img_out[i] = 0;
    }

    // Run HLS kernel
    warp_transform_accel(img_in, transform, img_out, HEIGHT, WIDTH);

    // Unpack output pixels
    uint8_t out_raw[NPIX];
    for (int i = 0; i < NWORDS; i++) {
        for (int b = 0; b < 4; b++) {
            out_raw[i*4 + b] = (uint8_t)(img_out[i].range(b*8+7, b*8).to_uint());
        }
    }

    f = fopen(argv[3], "wb");
    if (!f) { fprintf(stderr, "Cannot open output %s\n", argv[3]); return 1; }
    fwrite(out_raw, 1, NPIX, f); fclose(f);

    // Read expected
    uint8_t exp_raw[NPIX];
    f = fopen(argv[4], "rb");
    if (!f) { fprintf(stderr, "Cannot open expected %s\n", argv[4]); return 1; }
    fread(exp_raw, 1, NPIX, f); fclose(f);

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
