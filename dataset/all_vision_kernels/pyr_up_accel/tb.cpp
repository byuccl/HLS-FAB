#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_pyr_up_accel_config.h"

void pyr_up_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                  int in_rows, int in_cols, int out_rows, int out_cols);

static constexpr int NPIX_IN  = HEIGHT * WIDTH;
static constexpr int NPIX_OUT = 2 * HEIGHT * 2 * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <in.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    uint8_t in_raw[NPIX_IN];
    FILE* f = fopen(argv[1], "rb"); fread(in_raw, 1, NPIX_IN, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH>  img_inp[NPIX_IN];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NPIX_OUT];
    for (int i = 0; i < NPIX_IN; i++) img_inp[i] = in_raw[i];

    pyr_up_accel(img_inp, img_out, HEIGHT, WIDTH, 2 * HEIGHT, 2 * WIDTH);

    uint8_t out_raw[NPIX_OUT];
    for (int i = 0; i < NPIX_OUT; i++) out_raw[i] = (uint8_t)(img_out[i].to_uint());

    f = fopen(argv[2], "wb"); fwrite(out_raw, 1, NPIX_OUT, f); fclose(f);

    uint8_t exp_raw[NPIX_OUT];
    f = fopen(argv[3], "rb"); fread(exp_raw, 1, NPIX_OUT, f); fclose(f);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX_OUT; i++) {
        int d = abs((int)out_raw[i] - (int)exp_raw[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1) {
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n", i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>1): %d / %d\n", mismatches, NPIX_OUT);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
