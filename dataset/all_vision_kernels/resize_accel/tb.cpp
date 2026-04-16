#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_resize_accel_config.h"

void resize_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                  int rows_in, int cols_in, int rows_out, int cols_out);

static constexpr int NPIX_IN  = HEIGHT * WIDTH;
static constexpr int NPIX_OUT = NEWHEIGHT * NEWWIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <in.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    uint8_t in_raw[NPIX_IN], out_raw[NPIX_OUT], exp_raw[NPIX_OUT];
    FILE* f = fopen(argv[1], "rb"); fread(in_raw, 1, NPIX_IN, f); fclose(f);

    resize_accel((ap_uint<INPUT_PTR_WIDTH>*)in_raw,
                 (ap_uint<OUTPUT_PTR_WIDTH>*)out_raw,
                 HEIGHT, WIDTH, NEWHEIGHT, NEWWIDTH);

    f = fopen(argv[2], "wb"); fwrite(out_raw, 1, NPIX_OUT, f); fclose(f);
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
