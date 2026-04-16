// tb.cpp for min_max_loc_accel
// Signature: void min_max_loc_accel(ap_uint<8>* img_in,
//                                    int32_t& min_value, int32_t& max_value,
//                                    uint16_t& min_loc_x, uint16_t& min_loc_y,
//                                    uint16_t& max_loc_x, uint16_t& max_loc_y,
//                                    int height, int width)
// INPUT_PTR_WIDTH=8; n_in=1, n_out=6 (min_value,max_value,min_loc_x,min_loc_y,max_loc_x,max_loc_y)
// HEIGHT=128, WIDTH=128
//
// argv[1]  = input_img_in.bin
// argv[2]  = output_min_value.bin  (4 bytes, int32)
// argv[3]  = output_max_value.bin  (4 bytes, int32)
// argv[4]  = output_min_loc_x.bin  (2 bytes, uint16)
// argv[5]  = output_min_loc_y.bin  (2 bytes, uint16)
// argv[6]  = output_max_loc_x.bin  (2 bytes, uint16)
// argv[7]  = output_max_loc_y.bin  (2 bytes, uint16)
// argv[8]  = expected_min_value.bin
// argv[9]  = expected_max_value.bin
// argv[10] = expected_min_loc_x.bin
// argv[11] = expected_min_loc_y.bin
// argv[12] = expected_max_loc_x.bin
// argv[13] = expected_max_loc_y.bin

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdint.h>
#include "ap_int.h"

#define HEIGHT 128
#define WIDTH  128
#define INPUT_PTR_WIDTH 8

// Forward-declare HLS top function
void min_max_loc_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                       int32_t& min_value,
                       int32_t& max_value,
                       uint16_t& min_loc_x,
                       uint16_t& min_loc_y,
                       uint16_t& max_loc_x,
                       uint16_t& max_loc_y,
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
    // argc = 1 + 1(in) + 6(out) + 6(exp) = 14
    if (argc < 14) {
        fprintf(stderr, "Usage: %s in_img_in out_minv out_maxv out_minx out_miny out_maxx out_maxy"
                        " exp_minv exp_maxv exp_minx exp_miny exp_maxx exp_maxy\n", argv[0]);
        return 1;
    }

    const int N = HEIGHT * WIDTH;

    ap_uint<INPUT_PTR_WIDTH> img_in[N];

    {
        uint8_t raw[N];
        read_bin(argv[1], raw, N);
        for (int i = 0; i < N; i++) img_in[i] = raw[i];
    }

    int32_t  min_value = 0, max_value = 0;
    uint16_t min_loc_x = 0, min_loc_y = 0;
    uint16_t max_loc_x = 0, max_loc_y = 0;

    min_max_loc_accel(img_in, min_value, max_value,
                      min_loc_x, min_loc_y, max_loc_x, max_loc_y,
                      HEIGHT, WIDTH);

    // Write outputs
    write_bin(argv[2],  &min_value, sizeof(int32_t));
    write_bin(argv[3],  &max_value, sizeof(int32_t));
    write_bin(argv[4],  &min_loc_x, sizeof(uint16_t));
    write_bin(argv[5],  &min_loc_y, sizeof(uint16_t));
    write_bin(argv[6],  &max_loc_x, sizeof(uint16_t));
    write_bin(argv[7],  &max_loc_y, sizeof(uint16_t));

    // Compare
    {
        int32_t  exp_minv, exp_maxv;
        uint16_t exp_minx, exp_miny, exp_maxx, exp_maxy;
        read_bin(argv[8],  &exp_minv, sizeof(int32_t));
        read_bin(argv[9],  &exp_maxv, sizeof(int32_t));
        read_bin(argv[10], &exp_minx, sizeof(uint16_t));
        read_bin(argv[11], &exp_miny, sizeof(uint16_t));
        read_bin(argv[12], &exp_maxx, sizeof(uint16_t));
        read_bin(argv[13], &exp_maxy, sizeof(uint16_t));

        int mismatches = 0;
        int64_t max_diff = 0;

        // Helper lambda-free comparisons
        {
            int64_t d = (int64_t)min_value - (int64_t)exp_minv;
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
            if (d > 0) mismatches++;
        }
        {
            int64_t d = (int64_t)max_value - (int64_t)exp_maxv;
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
            if (d > 0) mismatches++;
        }
        {
            int d = (int)min_loc_x - (int)exp_minx;
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
            if (d > 0) mismatches++;
        }
        {
            int d = (int)min_loc_y - (int)exp_miny;
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
            if (d > 0) mismatches++;
        }
        {
            int d = (int)max_loc_x - (int)exp_maxx;
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
            if (d > 0) mismatches++;
        }
        {
            int d = (int)max_loc_y - (int)exp_maxy;
            if (d < 0) d = -d;
            if (d > max_diff) max_diff = d;
            if (d > 0) mismatches++;
        }

        printf("Max diff: %lld\n", (long long)max_diff);
        printf("Mismatches: %d / 6\n", mismatches);
        if (mismatches == 0) { printf("PASS\n"); return 0; }
        else                  { printf("FAIL\n"); return 1; }
    }
}
