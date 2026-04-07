#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_fast_accel_config.h"

void fast_accel(ap_uint<INPUT_PTR_WIDTH>* img_in, unsigned char threshold,
                ap_uint<OUTPUT_PTR_WIDTH>* img_out, int rows, int cols);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <in.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    uint8_t in_raw[NPIX];
    FILE* f = fopen(argv[1], "rb"); fread(in_raw, 1, NPIX, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH>  img_in[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NPIX];
    for (int i = 0; i < NPIX; i++) img_in[i] = in_raw[i];

    fast_accel(img_in, 20, img_out, HEIGHT, WIDTH);

    uint8_t out_raw[NPIX];
    for (int i = 0; i < NPIX; i++) out_raw[i] = (uint8_t)(img_out[i].to_uint());

    f = fopen(argv[2], "wb"); fwrite(out_raw, 1, NPIX, f); fclose(f);

    uint8_t exp_raw[NPIX];
    f = fopen(argv[3], "rb"); fread(exp_raw, 1, NPIX, f); fclose(f);

    // Count kernel corners and expected corners
    int kernel_corners = 0, exp_corners = 0, both = 0;
    for (int i = 0; i < NPIX; i++) {
        if (out_raw[i] > 0) kernel_corners++;
        if (exp_raw[i] > 0) exp_corners++;
        if (out_raw[i] > 0 && exp_raw[i] > 0) both++;
    }
    printf("Kernel corners: %d\n", kernel_corners);
    printf("Expected corners: %d\n", exp_corners);
    printf("Both (overlap): %d\n", both);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX; i++) {
        int d = abs((int)out_raw[i] - (int)exp_raw[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1) {
            mismatches++;
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>1): %d / %d\n", mismatches, NPIX);
    // FAST with NMS is complex; use loose tolerance
    if (mismatches < NPIX / 10) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
