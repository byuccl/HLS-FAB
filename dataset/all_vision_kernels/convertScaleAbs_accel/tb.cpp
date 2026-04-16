// tb.cpp for convertScaleAbs_accel
// Signature: void convertScaleAbs_accel(ap_uint<8>* img_in, float scale, float shift,
//                                        ap_uint<8>* img_out, int height, int width)
// INPUT_PTR_WIDTH=8, OUTPUT_PTR_WIDTH=8
// scale=1.5, shift=10.0 (from generate.py)
//
// argv[1] = input_img_in.bin
// argv[2] = output_img_out.bin
// argv[3] = expected_img_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH  8
#define OUTPUT_PTR_WIDTH 8

static const float SCALE_VAL = 1.5f;
static const float SHIFT_VAL = 10.0f;

// Forward-declare HLS top function
void convertScaleAbs_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                           float scale,
                           float shift,
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
    if (argc < 4) { fprintf(stderr, "Usage: %s in_img_in out_img_out exp_img_out\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH>  img_in[N];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in[i] = raw[i];
    }

    convertScaleAbs_accel(img_in, SCALE_VAL, SHIFT_VAL, img_out, HEIGHT, WIDTH);

    {
        uint8_t raw[N];
        for (int i = 0; i < N; i++) raw[i] = (uint8_t)(img_out[i].to_uint());
        write_bin(argv[2], raw, N);
    }

    {
        uint8_t expected[N];
        read_bin(argv[3], expected, N);

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
