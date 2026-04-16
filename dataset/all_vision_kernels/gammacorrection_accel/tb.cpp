// tb.cpp for gammacorrection_accel
// Signature: void gammacorrection_accel(ap_uint<32>* img_inp, ap_uint<32>* img_out,
//                                        unsigned char gamma_lut[256*3], int rows, int cols)
// INPUT_PTR_WIDTH=32, OUTPUT_PTR_WIDTH=32; XF_8UC3 (3-channel), HEIGHT=128, WIDTH=128
// gamma_lut computed with GAMMA_VAL_R=0.5, GAMMA_VAL_G=0.8, GAMMA_VAL_B=0.8
//
// Packing: 3-channel image has H*W*3 bytes, packed 4 bytes per 32-bit word
// N_BYTES = 128*128*3 = 49152; N_WORDS = 49152/4 = 12288
//
// argv[1] = input_img_inp.bin  (49152 bytes)
// argv[2] = output_img_out.bin (49152 bytes)
// argv[3] = expected_img_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define CHANNELS 3
#define INPUT_PTR_WIDTH  32
#define OUTPUT_PTR_WIDTH 32

static const float GAMMA_VAL[3] = {0.5f, 0.8f, 0.8f};

static const int N_BYTES = HEIGHT * WIDTH * CHANNELS;
static const int BYTES_PER_WORD_IN  = INPUT_PTR_WIDTH  / 8;
static const int BYTES_PER_WORD_OUT = OUTPUT_PTR_WIDTH / 8;
static const int N_WORDS_IN  = (N_BYTES + BYTES_PER_WORD_IN  - 1) / BYTES_PER_WORD_IN;
static const int N_WORDS_OUT = (N_BYTES + BYTES_PER_WORD_OUT - 1) / BYTES_PER_WORD_OUT;

// Forward-declare HLS top function
void gammacorrection_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                           ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                           unsigned char gamma_lut[256 * 3],
                           int rows,
                           int cols);

// Compute LUT matching C++ testbench compute_gamma()
static void compute_gamma_lut(unsigned char lut[256 * 3]) {
    for (int ch = 0; ch < 3; ch++) {
        float inv_gamma = 1.0f / GAMMA_VAL[ch];
        for (int i = 0; i < 256; i++) {
            float norm = (float)i / 255.0f;
            float powval = powf(norm, inv_gamma);
            int16_t val = (int16_t)(powval * 255.0f);  // truncate toward zero
            if (val < 0)   val = 0;
            if (val > 255) val = 255;
            lut[ch * 256 + i] = (unsigned char)val;
        }
    }
}

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
    if (argc < 4) { fprintf(stderr, "Usage: %s in_img_inp out_img_out exp_img_out\n", argv[0]); return 1; }

    ap_uint<INPUT_PTR_WIDTH>  img_inp[N_WORDS_IN];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N_WORDS_OUT];
    memset(img_inp, 0, sizeof(img_inp));
    memset(img_out, 0, sizeof(img_out));

    // Read input and pack
    {
        uint8_t raw[N_BYTES];
        read_bin(argv[1], raw, N_BYTES);
        for (int b = 0; b < N_BYTES; b++) {
            int w = b / BYTES_PER_WORD_IN;
            int byte_pos = b % BYTES_PER_WORD_IN;
            img_inp[w].range(byte_pos * 8 + 7, byte_pos * 8) = raw[b];
        }
    }

    // Compute LUT
    unsigned char gamma_lut[256 * 3];
    compute_gamma_lut(gamma_lut);

    // Call kernel
    gammacorrection_accel(img_inp, img_out, gamma_lut, HEIGHT, WIDTH);

    // Unpack output and write
    {
        uint8_t raw[N_BYTES];
        for (int b = 0; b < N_BYTES; b++) {
            int w = b / BYTES_PER_WORD_OUT;
            int byte_pos = b % BYTES_PER_WORD_OUT;
            raw[b] = (uint8_t)(img_out[w].range(byte_pos * 8 + 7, byte_pos * 8).to_uint());
        }
        write_bin(argv[2], raw, N_BYTES);
    }

    // Compare
    {
        uint8_t expected[N_BYTES];
        read_bin(argv[3], expected, N_BYTES);

        int mismatches = 0;
        int max_diff = 0;
        for (int b = 0; b < N_BYTES; b++) {
            int w = b / BYTES_PER_WORD_OUT;
            int byte_pos = b % BYTES_PER_WORD_OUT;
            int got = (int)(img_out[w].range(byte_pos * 8 + 7, byte_pos * 8).to_uint());
            int exp = (int)expected[b];
            int diff = got - exp; if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }
        printf("Max diff: %d\n", max_diff);
        printf("Mismatches: %d / %d\n", mismatches, N_BYTES);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
