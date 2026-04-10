#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_dilation_accel_config.h"

void dilation_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                    unsigned char* kernel,
                    int height, int width);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <in.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    uint8_t in_raw[NPIX];
    FILE* f = fopen(argv[1], "rb"); fread(in_raw, 1, NPIX, f); fclose(f);

    // Cross-shaped 3x3 structuring element (XF_SHAPE_CROSS)
    // 0 1 0 / 1 1 1 / 0 1 0
    unsigned char se[9] = {0, 1, 0, 1, 1, 1, 0, 1, 0};

    ap_uint<INPUT_PTR_WIDTH>  img_inp[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NPIX];
    for (int i = 0; i < NPIX; i++) img_inp[i] = in_raw[i];

    dilation_accel(img_inp, img_out, se, HEIGHT, WIDTH);

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
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n", i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>1): %d / %d\n", mismatches, NPIX);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
