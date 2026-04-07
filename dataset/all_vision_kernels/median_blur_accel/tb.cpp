// tb.cpp for median_blur_accel
// Signature: void median_blur_accel(ap_uint<128>* img_in, int rows, int cols,
//                                    ap_uint<128>* img_out)
// INPUT_PTR_WIDTH=128, OUTPUT_PTR_WIDTH=128; uint8 pixels, 16 pixels per 128-bit word
// HEIGHT=128, WIDTH=128
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
#define INPUT_PTR_WIDTH  128
#define OUTPUT_PTR_WIDTH 128

// 16 uint8 pixels per 128-bit word
static const int PIX_PER_WORD = INPUT_PTR_WIDTH / 8;  // = 16
static const int N = HEIGHT * WIDTH;
static const int N_WORDS = N / PIX_PER_WORD;  // = 16384/16 = 1024

// Forward-declare HLS top function
void median_blur_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                       int rows,
                       int cols,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_out);

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

    ap_uint<INPUT_PTR_WIDTH>  img_in[N_WORDS];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N_WORDS];
    memset(img_in,  0, sizeof(img_in));
    memset(img_out, 0, sizeof(img_out));

    // Read and pack uint8 pixels
    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) {
            int w = i / PIX_PER_WORD;
            int p = i % PIX_PER_WORD;
            img_in[w].range(p * 8 + 7, p * 8) = raw[i];
        }
    }

    median_blur_accel(img_in, HEIGHT, WIDTH, img_out);

    // Unpack and write uint8 output
    {
        uint8_t raw[N];
        for (int i = 0; i < N; i++) {
            int w = i / PIX_PER_WORD;
            int p = i % PIX_PER_WORD;
            raw[i] = (uint8_t)(img_out[w].range(p * 8 + 7, p * 8).to_uint());
        }
        write_bin(argv[2], raw, N);
    }

    // Compare
    {
        uint8_t expected[N];
        read_bin(argv[3], expected, N);

        int mismatches = 0;
        int max_diff = 0;
        for (int i = 0; i < N; i++) {
            int w = i / PIX_PER_WORD;
            int p = i % PIX_PER_WORD;
            int got = (int)(img_out[w].range(p * 8 + 7, p * 8).to_uint());
            int exp = (int)expected[i];
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
