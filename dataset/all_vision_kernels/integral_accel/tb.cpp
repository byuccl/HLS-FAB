// tb.cpp for integral_accel
// Signature: void integral_accel(ap_uint<8>* img_inp, ap_uint<32>* img_out,
//                                  int rows, int cols)
// INPUT_PTR_WIDTH=8 (1 byte/word), OUTPUT_PTR_WIDTH=32 (4 bytes/word, int32)
// Height=128, Width=128
//
// argv[1] = input_img_inp.bin  (128*128 = 16384 bytes, uint8)
// argv[2] = output_img_out.bin (128*128*4 = 65536 bytes, int32 little-endian)
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

// Forward-declare HLS top function
void integral_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
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
    if (argc < 4) { fprintf(stderr, "Usage: %s in_img_inp out_img_out exp_img_out\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH>  img_inp[N];
    ap_uint<OUTPUT_PTR_WIDTH> img_out[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_inp[i] = raw[i];
    }

    integral_accel(img_inp, img_out, HEIGHT, WIDTH);

    // Write output as int32 little-endian
    {
        int32_t raw[N];
        for (int i = 0; i < N; i++) {
            // ap_uint<32> holds int32 bits; reinterpret
            uint32_t u = img_out[i].to_uint();
            raw[i] = (int32_t)u;
        }
        write_bin(argv[2], raw, N * sizeof(int32_t));
    }

    // Compare
    {
        int32_t expected[N];
        read_bin(argv[3], expected, N * sizeof(int32_t));

        int mismatches = 0;
        int64_t max_diff = 0;
        for (int i = 0; i < N; i++) {
            int32_t got = (int32_t)(img_out[i].to_uint());
            int32_t exp = expected[i];
            int64_t diff = (int64_t)got - (int64_t)exp;
            if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }
        printf("Max diff: %lld\n", (long long)max_diff);
        printf("Mismatches: %d / %d\n", mismatches, N);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
