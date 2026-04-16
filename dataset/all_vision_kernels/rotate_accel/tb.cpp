// tb.cpp for rotate_accel
// Signature: void rotate_accel(ap_uint<8>* img_in, ap_uint<8>* img_out,
//                               int height, int width, int direction)
// INPUT_PTR_WIDTH=8, OUTPUT_PTR_WIDTH=8; HEIGHT=512, WIDTH=512
// R_90=1 so DIRECTION=1 (90-degree clockwise)
// Output shape for 90-deg rotation: (WIDTH, HEIGHT) = (512, 512) - square so same size
//
// argv[1] = input_img_in.bin   (512*512 = 262144 bytes, uint8)
// argv[2] = output_img_out.bin (512*512 = 262144 bytes, uint8)
// argv[3] = expected_img_out.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 512
#define WIDTH  512
#define INPUT_PTR_WIDTH  8
#define OUTPUT_PTR_WIDTH 8
#define DIRECTION 1

// Forward-declare HLS top function
void rotate_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                  ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                  int height,
                  int width,
                  int direction);

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

    const int N_IN  = HEIGHT * WIDTH;
    const int N_OUT = WIDTH  * HEIGHT;  // same for square image

    // Allocate on heap for large arrays
    ap_uint<INPUT_PTR_WIDTH>*  img_in  = new ap_uint<INPUT_PTR_WIDTH>[N_IN];
    ap_uint<OUTPUT_PTR_WIDTH>* img_out = new ap_uint<OUTPUT_PTR_WIDTH>[N_OUT];

    {
        uint8_t* raw = new uint8_t[N_IN];
        read_bin(argv[1], raw, N_IN);
        for (int i = 0; i < N_IN; i++) img_in[i] = raw[i];
        delete[] raw;
    }

    rotate_accel(img_in, img_out, HEIGHT, WIDTH, DIRECTION);

    {
        uint8_t* raw = new uint8_t[N_OUT];
        for (int i = 0; i < N_OUT; i++) raw[i] = (uint8_t)(img_out[i].to_uint());
        write_bin(argv[2], raw, N_OUT);
        delete[] raw;
    }

    {
        uint8_t* expected = new uint8_t[N_OUT];
        read_bin(argv[3], expected, N_OUT);

        int mismatches = 0;
        int max_diff = 0;
        for (int i = 0; i < N_OUT; i++) {
            int got  = (int)(img_out[i].to_uint());
            int exp  = (int)expected[i];
            int diff = got - exp; if (diff < 0) diff = -diff;
            if (diff > max_diff) max_diff = diff;
            if (diff > 0) mismatches++;
        }
        printf("Max diff: %d\n", max_diff);
        printf("Mismatches: %d / %d\n", mismatches, N_OUT);
        delete[] expected;

        delete[] img_in;
        delete[] img_out;

        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
