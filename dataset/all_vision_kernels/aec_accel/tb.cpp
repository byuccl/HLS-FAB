// tb.cpp for aec_accel
// Signature: void aec_accel(ap_uint<64>* src, ap_uint<64>* dst, int rows, int cols)
// INPUT_PTR_WIDTH=64, OUTPUT_PTR_WIDTH=64; IN_TYPE=XF_8UC3 (3-channel BGR uint8)
// HEIGHT=128, WIDTH=128; N_BYTES = 128*128*3 = 49152; N_WORDS = 49152/8 = 6144
//
// argv[1] = input_src.bin    (49152 bytes, uint8 BGR)
// argv[2] = output_dst.bin   (49152 bytes, uint8 BGR)
// argv[3] = expected_dst.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define CHANNELS 3
#define INPUT_PTR_WIDTH  64
#define OUTPUT_PTR_WIDTH 64

static const int N_BYTES      = HEIGHT * WIDTH * CHANNELS;
static const int BYTES_PER_WORD_IN  = INPUT_PTR_WIDTH  / 8;
static const int BYTES_PER_WORD_OUT = OUTPUT_PTR_WIDTH / 8;
static const int N_WORDS_IN   = (N_BYTES + BYTES_PER_WORD_IN  - 1) / BYTES_PER_WORD_IN;
static const int N_WORDS_OUT  = (N_BYTES + BYTES_PER_WORD_OUT - 1) / BYTES_PER_WORD_OUT;

// Forward-declare HLS top function
void aec_accel(ap_uint<INPUT_PTR_WIDTH>* src,
               ap_uint<OUTPUT_PTR_WIDTH>* dst,
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
    // argc = 1 + 1(in) + 1(out) + 1(exp) = 4
    if (argc < 4) { fprintf(stderr, "Usage: %s in_src out_dst exp_dst\n", argv[0]); return 1; }

    ap_uint<INPUT_PTR_WIDTH>  src_words[N_WORDS_IN];
    ap_uint<OUTPUT_PTR_WIDTH> dst_words[N_WORDS_OUT];
    memset(src_words, 0, sizeof(src_words));
    memset(dst_words, 0, sizeof(dst_words));

    // Read input and pack into 64-bit words
    {
        uint8_t raw[N_BYTES];
        read_bin(argv[1], raw, N_BYTES);
        for (int b = 0; b < N_BYTES; b++) {
            int w = b / BYTES_PER_WORD_IN;
            int byte_pos = b % BYTES_PER_WORD_IN;
            src_words[w].range(byte_pos * 8 + 7, byte_pos * 8) = raw[b];
        }
    }

    // Call kernel
    aec_accel(src_words, dst_words, HEIGHT, WIDTH);

    // Unpack output and write
    {
        uint8_t raw[N_BYTES];
        for (int b = 0; b < N_BYTES; b++) {
            int w = b / BYTES_PER_WORD_OUT;
            int byte_pos = b % BYTES_PER_WORD_OUT;
            raw[b] = (uint8_t)(dst_words[w].range(byte_pos * 8 + 7, byte_pos * 8).to_uint());
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
            int got = (int)(dst_words[w].range(byte_pos * 8 + 7, byte_pos * 8).to_uint());
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
