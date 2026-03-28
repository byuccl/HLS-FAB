// tb.cpp for paintmask_accel
// Signature: void paintmask_accel(ap_uint<8>* img_in, ap_uint<8>* mask_in,
//                                   unsigned char* color, ap_uint<8>* img_out,
//                                   int height, int width)
// INPUT_PTR_WIDTH=8, OUTPUT_PTR_WIDTH=8; single-channel uint8
// COLOR=150 (from generate.py)
// HEIGHT=128, WIDTH=128
//
// argv[1] = input_img_in.bin
// argv[2] = input_mask_in.bin
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
#define OUTPUT_PTR_WIDTH 8

static const unsigned char COLOR_VAL = 150;

// Forward-declare HLS top function
void paintmask_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                     ap_uint<INPUT_PTR_WIDTH>* mask_in,
                     unsigned char* color,
                     ap_uint<OUTPUT_PTR_WIDTH>* img_out,
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
    // argc = 1 + 2(in) + 1(out) + 1(exp) = 5
    if (argc < 5) { fprintf(stderr, "Usage: %s in_img_in in_mask_in out_img_out exp_img_out\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH>  img_in[N];
    ap_uint<INPUT_PTR_WIDTH>  mask_in[N];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in[i] = raw[i];
        read_bin(argv[2], raw, N);
        for (int i = 0; i < N; i++) mask_in[i] = raw[i];
    }

    unsigned char color[1] = { COLOR_VAL };
    paintmask_accel(img_in, mask_in, color, img_out, HEIGHT, WIDTH);

    {
        uint8_t raw[N];
        for (int i = 0; i < N; i++) raw[i] = (uint8_t)(img_out[i].to_uint());
        write_bin(argv[3], raw, N);
    }

    {
        uint8_t expected[N];
        read_bin(argv[4], expected, N);

        int mismatches = 0;
        int max_diff = 0;
        for (int i = 0; i < N; i++) {
            int got  = (int)(img_out[i].to_uint());
            int exp  = (int)expected[i];
            int diff = got - exp; if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }
        printf("Max diff: %d\n", max_diff);
        printf("Mismatches: %d / %d\n", mismatches, N);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
