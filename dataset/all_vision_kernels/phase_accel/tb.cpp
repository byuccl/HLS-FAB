#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_phase_accel_config.h"

void phase_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp1,
                 ap_uint<INPUT_PTR_WIDTH>* img_inp2,
                 ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                 int rows, int cols);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <in1.bin> <in2.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    // Raw byte cast approach (matching official Vitis testbench: (ap_uint<N>*)mat.data)
    int16_t in1_raw[NPIX], in2_raw[NPIX], out_raw[NPIX], exp_raw[NPIX];

    FILE* f = fopen(argv[1], "rb"); fread(in1_raw, sizeof(int16_t), NPIX, f); fclose(f);
    f = fopen(argv[2], "rb"); fread(in2_raw, sizeof(int16_t), NPIX, f); fclose(f);

    phase_accel((ap_uint<INPUT_PTR_WIDTH>*)in1_raw,
                (ap_uint<INPUT_PTR_WIDTH>*)in2_raw,
                (ap_uint<OUTPUT_PTR_WIDTH>*)out_raw,
                HEIGHT, WIDTH);

    f = fopen(argv[3], "wb"); fwrite(out_raw, sizeof(int16_t), NPIX, f); fclose(f);
    f = fopen(argv[4], "rb"); fread(exp_raw, sizeof(int16_t), NPIX, f); fclose(f);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX; i++) {
        int d = abs((int)out_raw[i] - (int)exp_raw[i]);
        if (d > max_diff) max_diff = d;
        if (d > 100) {  // ~1.5 degree tolerance for fixed-point atan2 approximation
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n", i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>100): %d / %d\n", mismatches, NPIX);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
