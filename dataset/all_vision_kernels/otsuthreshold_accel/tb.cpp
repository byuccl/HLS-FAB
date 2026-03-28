// tb.cpp for otsuthreshold_accel
// Signature: void otsuthreshold_accel(ap_uint<8>* img_in, unsigned char& Otsuval,
//                                      int height, int width)
// INPUT_PTR_WIDTH=8; n_in=1, n_out=1 (Otsuval, scalar uint8)
// HEIGHT=128, WIDTH=128
//
// argv[1] = input_img_in.bin      (16384 bytes, uint8)
// argv[2] = output_Otsuval.bin    (1 byte, uint8)
// argv[3] = expected_Otsuval.bin  (1 byte, uint8)

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH 8

// Forward-declare HLS top function
void otsuthreshold_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                         unsigned char& Otsuval,
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
    // argc = 1 + 1(in) + 1(out) + 1(exp) = 4
    if (argc < 4) { fprintf(stderr, "Usage: %s in_img_in out_Otsuval exp_Otsuval\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH> img_in[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in[i] = raw[i];
    }

    unsigned char Otsuval = 0;
    otsuthreshold_accel(img_in, Otsuval, HEIGHT, WIDTH);

    write_bin(argv[2], &Otsuval, 1);

    {
        uint8_t expected = 0;
        read_bin(argv[3], &expected, 1);

        int diff = (int)Otsuval - (int)expected;
        if (diff < 0) diff = -diff;

        printf("Max diff: %d\n", diff);
        printf("Mismatches: %d / 1\n", (diff > 0) ? 1 : 0);
        if (diff == 0) { printf("PASS\n"); return 0; }
        else           { printf("FAIL\n"); return 1; }
    }
}
