#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include "xf_bilateral_filter_accel_config.h"

static const float SIGMA_COLOR = 10.0f;
static const float SIGMA_SPACE = 0.5f;

// Forward-declare the HLS top function exactly as it appears in the accel .cpp
void bilateral_filter_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                             float sigma_color,
                             float sigma_space,
                             int rows,
                             int cols,
                             ap_uint<OUTPUT_PTR_WIDTH>* img_out);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    uint8_t inp_raw[NPIX];
    FILE* f = fopen(argv[1], "rb");
    fread(inp_raw, 1, NPIX, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH>  img_in_hw[NPIX];
    ap_uint<OUTPUT_PTR_WIDTH> img_out_hw[NPIX];
    for (int i = 0; i < NPIX; i++) img_in_hw[i] = inp_raw[i];

    bilateral_filter_accel(img_in_hw, SIGMA_COLOR, SIGMA_SPACE, HEIGHT, WIDTH, img_out_hw);

    uint8_t out_raw[NPIX];
    for (int i = 0; i < NPIX; i++) out_raw[i] = (uint8_t)(img_out_hw[i].to_uint());

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
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n",
                       i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>1): %d / %d\n", mismatches, NPIX);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
