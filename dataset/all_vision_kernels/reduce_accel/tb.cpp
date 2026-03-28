// tb.cpp for reduce_accel
// Signature: void reduce_accel(ap_uint<8>* img_in, unsigned char dimension,
//                               ap_uint<32>* img_out, int height, int width)
// INPUT_PTR_WIDTH=8, OUTPUT_PTR_WIDTH=32; DIM=0 (reduce along rows -> 1 x WIDTH output)
// HEIGHT=128, WIDTH=128; output shape = (1, WIDTH) = 128 elements of int32
//
// argv[1] = input_img_in.bin   (128*128 = 16384 bytes, uint8)
// argv[2] = output_img_out.bin (128 * 4 = 512 bytes, int32)
// argv[3] = expected_img_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH  8
#define OUTPUT_PTR_WIDTH 32
#define DIM 0

static const int N_IN  = HEIGHT * WIDTH;
static const int N_OUT = WIDTH;  // DIM=0: 1 x WIDTH

// Forward-declare HLS top function
void reduce_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                  unsigned char dimension,
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

    ap_uint<INPUT_PTR_WIDTH>  img_in[N_IN];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N_OUT];

    {
        uint8_t raw[N_IN];
        read_bin(argv[1], raw, N_IN);
        for (int i = 0; i < N_IN; i++) img_in[i] = raw[i];
    }

    reduce_accel(img_in, (unsigned char)DIM, img_out, HEIGHT, WIDTH);

    // Write int32 output
    {
        int32_t raw[N_OUT];
        for (int i = 0; i < N_OUT; i++) raw[i] = (int32_t)(img_out[i].to_uint());
        write_bin(argv[2], raw, N_OUT * sizeof(int32_t));
    }

    // Compare
    {
        int32_t expected[N_OUT];
        read_bin(argv[3], expected, N_OUT * sizeof(int32_t));

        int mismatches = 0;
        int64_t max_diff = 0;
        for (int i = 0; i < N_OUT; i++) {
            int32_t got = (int32_t)(img_out[i].to_uint());
            int32_t exp = expected[i];
            int64_t diff = (int64_t)got - (int64_t)exp;
            if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }
        printf("Max diff: %lld\n", (long long)max_diff);
        printf("Mismatches: %d / %d\n", mismatches, N_OUT);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
