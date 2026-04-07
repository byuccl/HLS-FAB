#include <cstdio>
#include <cstdint>
#include "xf_sobel_accel_config.h"

// Forward-declare the HLS top function exactly as it appears in the accel .cpp
void sobel_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out1,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out2,
                 int rows, int cols);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <input.bin> <output_gx.bin> <output_gy.bin> <expected_gx.bin> <expected_gy.bin>\n", argv[0]);
        return 1;
    }

    uint8_t inp_raw[NPIX];
    FILE* f = fopen(argv[1], "rb");
    fread(inp_raw, 1, NPIX, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH>  img_inp[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out1[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out2[NPIX];
    for (int i = 0; i < NPIX; i++) img_inp[i] = inp_raw[i];

    sobel_accel(img_inp, img_out1, img_out2, HEIGHT, WIDTH);

    uint8_t out_gx[NPIX], out_gy[NPIX];
    for (int i = 0; i < NPIX; i++) {
        out_gx[i] = (uint8_t)(img_out1[i].to_uint());
        out_gy[i] = (uint8_t)(img_out2[i].to_uint());
    }

    f = fopen(argv[2], "wb"); fwrite(out_gx, 1, NPIX, f); fclose(f);
    f = fopen(argv[3], "wb"); fwrite(out_gy, 1, NPIX, f); fclose(f);

    uint8_t exp_gx[NPIX], exp_gy[NPIX];
    f = fopen(argv[4], "rb"); fread(exp_gx, 1, NPIX, f); fclose(f);
    f = fopen(argv[5], "rb"); fread(exp_gy, 1, NPIX, f); fclose(f);

    int mismatches_gx = 0, max_diff_gx = 0;
    int mismatches_gy = 0, max_diff_gy = 0;

    // Check Gx
    for (int i = 0; i < NPIX; i++) {
        int d = abs((int)out_gx[i] - (int)exp_gx[i]);
        if (d > max_diff_gx) max_diff_gx = d;
        if (d > 1) {
            mismatches_gx++;
            if (mismatches_gx <= 5)
                printf("  mismatch Gx pixel %d: kernel=%d expected=%d diff=%d\n",
                       i, out_gx[i], exp_gx[i], d);
        }
    }

    for (int i = 0; i < NPIX; i++) {
        int d = abs((int)out_gy[i] - (int)exp_gy[i]);
        if (d > max_diff_gy) max_diff_gy = d;
        if (d > 1) {
            mismatches_gy++;
            if (mismatches_gy <= 5)
                printf("  mismatch Gy pixel %d: kernel=%d expected=%d diff=%d\n",
                       i, out_gy[i], exp_gy[i], d);
        }
    }

    printf("Max diff Gx: %d\n", max_diff_gx);
    printf("Mismatches Gx (diff>1): %d / %d\n", mismatches_gx, NPIX);
    printf("Max diff Gy: %d\n", max_diff_gy);
    printf("Mismatches Gy (diff>1): %d / %d\n", mismatches_gy, NPIX);

    if (mismatches_gx == 0 && mismatches_gy == 0) {
        printf("PASS\n");
        return 0;
    } else {
        printf("FAIL\n");
        return 1;
    }
}
