#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_histogram_accel_config.h"

void histogram_accel(ap_uint<INPUT_PTR_WIDTH>* img_in, unsigned int* histogram, int rows, int cols);

static constexpr int NPIX = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <in.bin> <out.bin> <expected.bin>\n", argv[0]);
        return 1;
    }
    uint8_t in_raw[NPIX];
    FILE* f = fopen(argv[1], "rb"); fread(in_raw, 1, NPIX, f); fclose(f);

    ap_uint<INPUT_PTR_WIDTH> img_in[NPIX];
    for (int i = 0; i < NPIX; i++) img_in[i] = in_raw[i];

    unsigned int histogram[256] = {0};
    histogram_accel(img_in, histogram, HEIGHT, WIDTH);

    f = fopen(argv[2], "wb"); fwrite(histogram, sizeof(unsigned int), 256, f); fclose(f);

    unsigned int exp_hist[256];
    f = fopen(argv[3], "rb"); fread(exp_hist, sizeof(unsigned int), 256, f); fclose(f);

    int mismatches = 0;
    unsigned int max_diff = 0;
    for (int i = 0; i < 256; i++) {
        int d = abs((int)histogram[i] - (int)exp_hist[i]);
        if ((unsigned int)d > max_diff) max_diff = d;
        if (d > 1) {
            mismatches++;
            if (mismatches <= 5)
                printf("  mismatch bin %d: kernel=%u expected=%u diff=%d\n", i, histogram[i], exp_hist[i], d);
        }
    }
    printf("Max diff: %u\n", max_diff);
    printf("Mismatches (diff>1): %d / 256\n", mismatches);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
