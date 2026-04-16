// tb.cpp for accumulate_squared
// argv[1] = input_img_in1.bin
// argv[2] = input_img_in2.bin
// argv[3] = output_img_out.bin
// argv[4] = expected_img_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH  8
#define OUTPUT_PTR_WIDTH 16

// Forward-declare HLS top function
void accumulate_squared(ap_uint<INPUT_PTR_WIDTH>* img_in1,
                        ap_uint<INPUT_PTR_WIDTH>* img_in2,
                        ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                        int height,
                        int width);

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
    // argc = 1 + 2 (in) + 1 (out) + 1 (exp) = 5
    if (argc < 5) { fprintf(stderr, "Usage: %s in1 in2 out exp\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH>  img_in1[N];
    ap_uint<INPUT_PTR_WIDTH>  img_in2[N];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N];

    // Read inputs
    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in1[i] = raw[i];
        read_bin(argv[2], raw, N);
        for (int i = 0; i < N; i++) img_in2[i] = raw[i];
    }

    // Call kernel
    accumulate_squared(img_in1, img_in2, img_out, HEIGHT, WIDTH);

    // Write output (uint16 little-endian)
    {
        uint16_t raw[N];
        for (int i = 0; i < N; i++) raw[i] = (uint16_t)(img_out[i].to_uint());
        write_bin(argv[3], raw, N * sizeof(uint16_t));
    }

    // Compare
    {
        uint16_t expected[N];
        read_bin(argv[4], expected, N * sizeof(uint16_t));

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
