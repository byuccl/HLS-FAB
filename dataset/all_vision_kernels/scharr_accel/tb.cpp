// tb.cpp for scharr_accel
// Signature: void scharr_accel(ap_uint<8>* img_inp, ap_uint<8>* img_out1,
//                               ap_uint<8>* img_out2, int rows, int cols)
// INPUT_PTR_WIDTH=8, OUTPUT_PTR_WIDTH=8; IN_TYPE=XF_8UC1, OUT_TYPE=XF_8UC1
// n_in=1, n_out=2 (img_out1=Gx, img_out2=Gy)
// HEIGHT=128, WIDTH=128
//
// argv[1] = input_img_inp.bin
// argv[2] = output_img_out1.bin (Gx)
// argv[3] = output_img_out2.bin (Gy)
// argv[4] = expected_img_out1.bin
// argv[5] = expected_img_out2.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH  8
#define OUTPUT_PTR_WIDTH 8

// Forward-declare HLS top function
void scharr_accel(ap_uint<INPUT_PTR_WIDTH>* img_inp,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out1,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out2,
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
    // argc = 1 + 1(in) + 2(out) + 2(exp) = 6
    if (argc < 6) { fprintf(stderr, "Usage: %s in_img_inp out_out1 out_out2 exp_out1 exp_out2\n", argv[0]); return 1; }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH>  img_inp[N];
    ap_uint<OUTPUT_PTR_WIDTH> img_out1[N];
    ap_uint<OUTPUT_PTR_WIDTH> img_out2[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_inp[i] = raw[i];
    }

    scharr_accel(img_inp, img_out1, img_out2, HEIGHT, WIDTH);

    {
        uint8_t raw[N];
        for (int i = 0; i < N; i++) raw[i] = (uint8_t)(img_out1[i].to_uint());
        write_bin(argv[2], raw, N);
        for (int i = 0; i < N; i++) raw[i] = (uint8_t)(img_out2[i].to_uint());
        write_bin(argv[3], raw, N);
    }

    // Compare both outputs
    {
        uint8_t exp1[N], exp2[N];
        read_bin(argv[4], exp1, N);
        read_bin(argv[5], exp2, N);

        int mismatches = 0;
        int max_diff = 0;

        for (int i = 0; i < N; i++) {
            int got  = (int)(img_out1[i].to_uint());
            int exp  = (int)exp1[i];
            int diff = got - exp; if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }
        for (int i = 0; i < N; i++) {
            int got  = (int)(img_out2[i].to_uint());
            int exp  = (int)exp2[i];
            int diff = got - exp; if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }

        printf("Max diff: %d\n", max_diff);
        printf("Mismatches: %d / %d\n", mismatches, N * 2);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
