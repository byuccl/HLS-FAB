#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "xf_clahe_accel_config.h"

// Forward declaration matching xf_clahe_accel.cpp exactly
extern "C" {
void clahe_accel(ap_uint<INPUT_PTR_WIDTH>* in_ptr,
                 ap_uint<OUTPUT_PTR_WIDTH>* out_ptr,
                 int height,
                 int width,
                 int clip,
                 int tilesY,
                 int tilesX);
}

static constexpr int NPIX = HEIGHT * WIDTH;
// 512-bit words, 64 bytes each -> NPIX/64 words
static constexpr int NWORDS = (NPIX + 63) / 64;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    // Read raw uint8 input
    static uint8_t inp_raw[NPIX];
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fread(inp_raw, 1, NPIX, f);
    fclose(f);

    // Pack into 512-bit words (64 pixels per word)
    static ap_uint<INPUT_PTR_WIDTH>  img_in_hw[NWORDS];
    static ap_uint<OUTPUT_PTR_WIDTH> img_out_hw[NWORDS];

    for (int w = 0; w < NWORDS; w++) {
        ap_uint<512> word = 0;
        for (int b = 0; b < 64; b++) {
            int idx = w * 64 + b;
            uint8_t pix = (idx < NPIX) ? inp_raw[idx] : 0;
            word.range((b+1)*8 - 1, b*8) = pix;
        }
        img_in_hw[w] = word;
    }

    // First call: flag=false -> writes LUT to _lut1, reads _lut2 (zeros) -> output zeros
    clahe_accel(img_in_hw, img_out_hw, HEIGHT, WIDTH, 40, TILES_X_MAX, TILES_Y_MAX);

    // Second call: flag=true -> writes LUT to _lut2, reads _lut1 (computed from call 1)
    clahe_accel(img_in_hw, img_out_hw, HEIGHT, WIDTH, 40, TILES_X_MAX, TILES_Y_MAX);

    // Unpack output
    static uint8_t out_raw[NPIX];
    for (int w = 0; w < NWORDS; w++) {
        for (int b = 0; b < 64; b++) {
            int idx = w * 64 + b;
            if (idx < NPIX) {
                out_raw[idx] = (uint8_t)(img_out_hw[w].range((b+1)*8 - 1, b*8).to_uint());
            }
        }
    }

    // Write output
    f = fopen(argv[2], "wb");
    fwrite(out_raw, 1, NPIX, f);
    fclose(f);

    // Load expected
    static uint8_t exp_raw[NPIX];
    f = fopen(argv[3], "rb");
    fread(exp_raw, 1, NPIX, f);
    fclose(f);

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
