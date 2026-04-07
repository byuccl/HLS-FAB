// tb.cpp for sum_accel
// Signature: void sum_accel(ap_uint<8>* img_in, double* sum_out)
// INPUT_PTR_WIDTH=8; output is double* (1 element)
// The generate.py models output as ap_uint<64>* (float64) but accel uses double*
// HEIGHT=128, WIDTH=128
//
// argv[1] = input_img_in.bin   (16384 bytes, uint8)
// argv[2] = output_sum_out.bin (8 bytes, float64/double)
// argv[3] = expected_sum_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include <math.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH 8

// Forward-declare HLS top function
void sum_accel(ap_uint<INPUT_PTR_WIDTH>* img_in, double* sum_out);

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
    // argc = 1 + 1(in) + 1(out) + 1(exp) = 4
    if (argc < 4) { fprintf(stderr, "Usage: %s in_img_in out_sum_out exp_sum_out\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH> img_in[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in[i] = raw[i];
    }

    double sum_out = 0.0;
    sum_accel(img_in, &sum_out);

    // Write as double (8 bytes, IEEE 754)
    write_bin(argv[2], &sum_out, sizeof(double));

    // Compare
    {
        double expected = 0.0;
        read_bin(argv[3], &expected, sizeof(double));

        double diff = sum_out - expected;
        if (diff < 0.0) diff = -diff;
        int mismatch = (diff > 0.5) ? 1 : 0;  // exact match expected for integer sum

        printf("Max diff: %g\n", diff);
        printf("Mismatches: %d / 1\n", mismatch);
        if (mismatch == 0) { printf("PASS\n"); return 0; }
        else               { printf("FAIL\n"); return 1; }
    }
}
