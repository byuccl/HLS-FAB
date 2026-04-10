// tb.cpp for magnitude_accel
// Signature: void magnitude_accel(ap_uint<128>* img_inp1, ap_uint<128>* img_inp2,
//                                   ap_uint<128>* img_out, int rows, int cols)
// INPUT_PTR_WIDTH=128 (16 bytes = 8 x int16 pixels per word)
// OUTPUT_PTR_WIDTH=128; IN_TYPE=XF_16SC1, OUT_TYPE=XF_16SC1
// HEIGHT=128, WIDTH=128
//
// argv[1] = input_img_inp1.bin  (128*128*2 bytes, int16)
// argv[2] = input_img_inp2.bin  (128*128*2 bytes, int16)
// argv[3] = output_img_out.bin  (128*128*2 bytes, int16)
// argv[4] = expected_img_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH  128
#define OUTPUT_PTR_WIDTH 128

// int16: 2 bytes each; pixels per word = 128/16 = 8
static const int PIX_PER_WORD = INPUT_PTR_WIDTH / 16;  // = 8
static const int N = HEIGHT * WIDTH;
static const int N_WORDS = N / PIX_PER_WORD;  // = 16384/8 = 2048

// Forward-declare HLS top function
void magnitude_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp1,
                     ap_uint<INPUT_PTR_WIDTH>* img_inp2,
                     ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                     int rows,
                     int cols);

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
    if (argc < 5) { fprintf(stderr, "Usage: %s in1 in2 out exp\n", argv[0]); return 1; }

    ap_uint<INPUT_PTR_WIDTH>  img_inp1[N_WORDS];
    ap_uint<INPUT_PTR_WIDTH>  img_inp2[N_WORDS];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N_WORDS];
    memset(img_inp1, 0, sizeof(img_inp1));
    memset(img_inp2, 0, sizeof(img_inp2));
    memset(img_out,  0, sizeof(img_out));

    // Read and pack int16 pixels into 128-bit words
    {
        int16_t raw[N];
        read_bin(argv[1], raw, N * sizeof(int16_t));
        for (int i = 0; i < N; i++) {
            int w = i / PIX_PER_WORD;
            int p = i % PIX_PER_WORD;
            img_inp1[w].range(p * 16 + 15, p * 16) = (uint16_t)raw[i];
        }
        read_bin(argv[2], raw, N * sizeof(int16_t));
        for (int i = 0; i < N; i++) {
            int w = i / PIX_PER_WORD;
            int p = i % PIX_PER_WORD;
            img_inp2[w].range(p * 16 + 15, p * 16) = (uint16_t)raw[i];
        }
    }

    magnitude_accel(img_inp1, img_inp2, img_out, HEIGHT, WIDTH);

    // Unpack and write int16 output
    {
        int16_t raw[N];
        for (int i = 0; i < N; i++) {
            int w = i / PIX_PER_WORD;
            int p = i % PIX_PER_WORD;
            raw[i] = (int16_t)(img_out[w].range(p * 16 + 15, p * 16).to_uint());
        }
        write_bin(argv[3], raw, N * sizeof(int16_t));
    }

    // Compare
    {
        int16_t expected[N];
        read_bin(argv[4], expected, N * sizeof(int16_t));

        int mismatches = 0;
        int max_diff = 0;
        for (int i = 0; i < N; i++) {
            int w = i / PIX_PER_WORD;
            int p = i % PIX_PER_WORD;
            int got = (int)(int16_t)(img_out[w].range(p * 16 + 15, p * 16).to_uint());
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
