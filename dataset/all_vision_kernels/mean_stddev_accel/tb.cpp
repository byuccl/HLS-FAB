// tb.cpp for mean_stddev_accel
// Signature: void mean_stddev_accel(ap_uint<8>* img_in, unsigned short* mean,
//                                    unsigned short* stddev, int height, int width)
// INPUT_PTR_WIDTH=8; outputs are scalar unsigned short (uint16) pointers
// HEIGHT=128, WIDTH=128
//
// n_in=1 (img_in), n_out=2 (mean, stddev)
// argv[1] = input_img_in.bin    (16384 bytes, uint8)
// argv[2] = output_mean.bin     (2 bytes, uint16)
// argv[3] = output_stddev.bin   (2 bytes, uint16)
// argv[4] = expected_mean.bin   (2 bytes, uint16)
// argv[5] = expected_stddev.bin (2 bytes, uint16)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH  8

// Forward-declare HLS top function
void mean_stddev_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                       unsigned short* mean,
                       unsigned short* stddev,
                       int height,
                       int width);

static void read_bin(const char* path, void* buf, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    size_t n = fread(buf, 1, bytes, f);
    if (n != bytes) { fprintf(stderr, "Short read %s\n", path); exit(1); }
    fclose(f);
}

static void write_bin(const char* path, const void* buf, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for write\n", path); exit(1); }
    fwrite(buf, 1, bytes, f);
    fclose(f);
}

int main(int argc, char** argv) {
    // argc = 1 + 1(in) + 2(out) + 2(exp) = 6
    if (argc < 6) { fprintf(stderr, "Usage: %s in_img_in out_mean out_stddev exp_mean exp_stddev\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH> img_in[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in[i] = raw[i];
    }

    unsigned short mean_val   = 0;
    unsigned short stddev_val = 0;

    mean_stddev_accel(img_in, &mean_val, &stddev_val, HEIGHT, WIDTH);

    // Write outputs (uint16 LE)
    write_bin(argv[2], &mean_val,   sizeof(uint16_t));
    write_bin(argv[3], &stddev_val, sizeof(uint16_t));

    // Compare
    {
        uint16_t exp_mean, exp_stddev;
        read_bin(argv[4], &exp_mean,   sizeof(uint16_t));
        read_bin(argv[5], &exp_stddev, sizeof(uint16_t));

        int mismatches = 0;
        int max_diff = 0;

        int d1 = (int)mean_val - (int)exp_mean;
        if (d1 < 0) d1 = -d1;
        if (d1 > max_diff) max_diff = d1;
        if (d1 > 0) mismatches++;

        int d2 = (int)stddev_val - (int)exp_stddev;
        if (d2 < 0) d2 = -d2;
        if (d2 > max_diff) max_diff = d2;
        if (d2 > 0) mismatches++;

        printf("Max diff: %d\n", max_diff);
        printf("Mismatches: %d / 2\n", mismatches);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
