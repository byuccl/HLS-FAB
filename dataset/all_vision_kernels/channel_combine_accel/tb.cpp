#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_channel_combine_accel_config.h"

#if TWO_INPUT
void channel_combine_accel(ap_uint<INPUT_PTR_WIDTH>* img_in1,
                           ap_uint<INPUT_PTR_WIDTH>* img_in2,
                           ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                           int height, int width);
#endif

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <in1.bin> <in2.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    // Raw byte cast approach: INPUT_PTR_WIDTH=16 packs 2 uint8 pixels per word
    uint8_t in1_raw[NPIX], in2_raw[NPIX];
    uint16_t out_raw[NPIX], exp_raw[NPIX];

    FILE* f = fopen(argv[1], "rb"); fread(in1_raw, 1, NPIX, f); fclose(f);
    f = fopen(argv[2], "rb"); fread(in2_raw, 1, NPIX, f); fclose(f);

    channel_combine_accel((ap_uint<INPUT_PTR_WIDTH>*)in1_raw,
                          (ap_uint<INPUT_PTR_WIDTH>*)in2_raw,
                          (ap_uint<OUTPUT_PTR_WIDTH>*)out_raw,
                          HEIGHT, WIDTH);

    f = fopen(argv[3], "wb"); fwrite(out_raw, sizeof(uint16_t), NPIX, f); fclose(f);
    f = fopen(argv[4], "rb"); fread(exp_raw, sizeof(uint16_t), NPIX, f); fclose(f);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX; i++) {
        int d = abs((int)out_raw[i] - (int)exp_raw[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0) {
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch pixel %d: kernel=0x%04x expected=0x%04x diff=%d\n", i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>0): %d / %d\n", mismatches, NPIX);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
