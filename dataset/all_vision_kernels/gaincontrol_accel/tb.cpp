#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_gaincontrol_accel_config.h"

// INPUT_PTR_WIDTH=256, OUTPUT_PTR_WIDTH=256: 32 pixels per 256-bit word
static constexpr int NPIX = HEIGHT * WIDTH;
static constexpr int NWORDS = NPIX / (INPUT_PTR_WIDTH / 8);  // 16384 / 32 = 512

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    uint8_t inp_raw[NPIX];
    FILE* f = fopen(argv[1], "rb");
    fread(inp_raw, 1, NPIX, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH>  img_inp[NWORDS];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[NWORDS];

    // Pack 32 uint8 pixels per 256-bit word
    for (int w = 0; w < NWORDS; w++) {
        img_inp[w] = 0;
        for (int b = 0; b < 32; b++) {
            img_inp[w].range(b*8+7, b*8) = inp_raw[w*32 + b];
        }
        img_out[w] = 0;
    }

    // XF_BAYER_RG = 3 (BG=0, GB=1, GR=2, RG=3)
    gaincontrol_accel(img_inp, img_out, HEIGHT, WIDTH,
                      154, 140, 200, 3);

    // Unpack output
    uint8_t out_raw[NPIX];
    for (int w = 0; w < NWORDS; w++) {
        for (int b = 0; b < 32; b++) {
            out_raw[w*32 + b] = (uint8_t)(img_out[w].range(b*8+7, b*8).to_uint());
        }
    }

    f = fopen(argv[2], "wb"); fwrite(out_raw, 1, NPIX, f); fclose(f);

    uint8_t exp_raw[NPIX];
    f = fopen(argv[3], "rb"); fread(exp_raw, 1, NPIX, f); fclose(f);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX; i++) {
        int d = abs((int)out_raw[i] - (int)exp_raw[i]);
        if (d > max_diff) max_diff = d;
        if (d > 0) {
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n",
                       i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches: %d / %d\n", mismatches, NPIX);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
