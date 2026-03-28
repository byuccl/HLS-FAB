#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_lut_accel_config.h"

void lut_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
               unsigned char* lut_ptr,
               ap_uint<OUTPUT_PTR_WIDTH>* img_out,
               int height, int width);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <in.bin> <lut.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    uint8_t in_raw[NPIX];
    FILE* f = fopen(argv[1], "rb"); fread(in_raw, 1, NPIX, f); fclose(f);

    unsigned char lut_data[256];
    f = fopen(argv[2], "rb"); fread(lut_data, 1, 256, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH> img_in[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NPIX];
    for (int i = 0; i < NPIX; i++) img_in[i] = in_raw[i];

    lut_accel(img_in, lut_data, img_out, HEIGHT, WIDTH);

    uint8_t out_raw[NPIX];
    for (int i = 0; i < NPIX; i++) out_raw[i] = (uint8_t)(img_out[i].to_uint());

    f = fopen(argv[3], "wb"); fwrite(out_raw, 1, NPIX, f); fclose(f);

    uint8_t exp_raw[NPIX];
    f = fopen(argv[4], "rb"); fread(exp_raw, 1, NPIX, f); fclose(f);

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
