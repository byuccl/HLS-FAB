// tb.cpp for Filter2d_accel
// argv[1] = input_img_in.bin
// argv[2] = output_img_out.bin
// argv[3] = expected_img_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

// Constants
#define HEIGHT 128
#define WIDTH  128
#define FILTER_HEIGHT 3
#define FILTER_WIDTH  3
#define SHIFT_VAL     15
#define FILTER_COEFF  3640
#define INPUT_PTR_WIDTH  8
#define OUTPUT_PTR_WIDTH 8

// Forward-declare HLS top function
void Filter2d_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                    short int* filter,
                    unsigned char shift,
                    ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                    int rows,
                    int cols);

static void read_bin(const char* path, void* buf, size_t bytes) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    size_t n = fread(buf, 1, bytes, f);
    if (n != bytes) { fprintf(stderr, "Short read %s: got %zu expected %zu\n", path, n, bytes); exit(1); }
    fclose(f);
}

static void write_bin(const char* path, const void* buf, size_t bytes) {
    FILE* f = fopen(path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for write\n", path); exit(1); }
    fwrite(buf, 1, bytes, f);
    fclose(f);
}

int main(int argc, char** argv) {
    // argc = 1 + 1 (in) + 1 (out) + 1 (exp) = 4
    if (argc < 4) { fprintf(stderr, "Usage: %s in_img_in out_img_out exp_img_out\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    // Allocate arrays
    ap_uint<INPUT_PTR_WIDTH>  img_in[N];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N];

    // Read input
    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in[i] = raw[i];
    }

    // Build filter (all coefficients = 3640)
    short int filter[FILTER_HEIGHT * FILTER_WIDTH];
    for (int i = 0; i < FILTER_HEIGHT * FILTER_WIDTH; i++) filter[i] = (short int)FILTER_COEFF;

    // Call kernel
    Filter2d_accel(img_in, filter, (unsigned char)SHIFT_VAL, img_out, HEIGHT, WIDTH);

    // Write output
    {
        uint8_t raw[N];
        for (int i = 0; i < N; i++) raw[i] = (uint8_t)(img_out[i].to_uint());
        write_bin(argv[2], raw, N);
    }

    // Compare
    {
        uint8_t expected[N];
        read_bin(argv[3], expected, N);

        int mismatches = 0;
        int max_diff = 0;
        for (int i = 0; i < N; i++) {
            int got = (int)(img_out[i].to_uint());
            int exp = (int)expected[i];
            int diff = got - exp;
            if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }
        printf("Max diff: %d\n", max_diff);
        printf("Mismatches: %d / %d\n", mismatches, N);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
