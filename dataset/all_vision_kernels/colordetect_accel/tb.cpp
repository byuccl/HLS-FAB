#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include "xf_colordetect_accel_config.h"

// Forward-declare the HLS top function exactly as it appears in the accel .cpp
void colordetect_accel(ap_uint<INPUT_PTR_WIDTH>* img_in,
                       unsigned char* low_thresh,
                       unsigned char* high_thresh,
                       unsigned char* process_shape,
                       ap_uint<OUTPUT_PTR_WIDTH>* img_out,
                       int rows,
                       int cols);

static constexpr int NPIX_IN  = HEIGHT * WIDTH * 3;  // 3 bytes per BGR pixel
static constexpr int NPIX_OUT = HEIGHT * WIDTH;

int main(int argc, char** argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input.bin> <output.bin> <expected.bin>\n", argv[0]);
        return 1;
    }

    // Read raw BGR input (flat uint8, 3 bytes per pixel)
    uint8_t inp_raw[NPIX_IN];
    FILE* f = fopen(argv[1], "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", argv[1]); return 1; }
    fread(inp_raw, 1, NPIX_IN, f);
    fclose(f);

    // INPUT_PTR_WIDTH=32 means 4 bytes per AXI transfer.
    // The accel uses Array2xfMat with INPUT_PTR_WIDTH=32 and XF_8UC3,
    // so the input array depth = HEIGHT*WIDTH*3/4 elements of ap_uint<32>.
    // We must pack BGR bytes into 32-bit words (little-endian, 4 bytes per word,
    // last word of each pixel group may be partial).
    // Actually: XF_PIXELWIDTH(XF_8UC3, XF_NPPC1) = 24 bits per pixel.
    // Array2xfMat reads INPUT_PTR_WIDTH=32 bits at a time.
    // Depth = (HEIGHT*WIDTH*24/8) / (32/8) = HEIGHT*WIDTH*3/4
    // The testbench passes in_img.data directly cast to ap_uint<32>*,
    // which means memory is laid out as raw BGR bytes packed into 32-bit words.
    static constexpr int ARRAY_DEPTH_IN = (HEIGHT * WIDTH * 3 + 3) / 4;
    ap_uint<32> img_inp[ARRAY_DEPTH_IN];
    // Zero out padding
    for (int i = 0; i < ARRAY_DEPTH_IN; i++) img_inp[i] = 0;
    // Copy raw bytes into ap_uint<32> array (same byte layout as C cast)
    for (int i = 0; i < NPIX_IN; i++) {
        int word_idx = i / 4;
        int byte_off = i % 4;
        img_inp[word_idx].range(byte_off*8 + 7, byte_off*8) = inp_raw[i];
    }

    ap_uint<OUTPUT_PTR_WIDTH> img_out[NPIX_OUT];

    // Thresholds from testbench
    unsigned char low_thresh[9]  = {22,150,60, 38,150,60, 160,150,60};
    unsigned char high_thresh[9] = {38,255,255, 75,255,255, 179,255,255};
    unsigned char process_shape[9] = {1,1,1, 1,1,1, 1,1,1};

    colordetect_accel(img_inp, low_thresh, high_thresh, process_shape,
                      img_out, HEIGHT, WIDTH);

    // Write output
    uint8_t out_raw[NPIX_OUT];
    for (int i = 0; i < NPIX_OUT; i++) out_raw[i] = (uint8_t)(img_out[i].to_uint());

    f = fopen(argv[2], "wb");
    fwrite(out_raw, 1, NPIX_OUT, f);
    fclose(f);

    // Read expected
    uint8_t exp_raw[NPIX_OUT];
    f = fopen(argv[3], "rb");
    fread(exp_raw, 1, NPIX_OUT, f);
    fclose(f);

    int mismatches = 0, max_diff = 0;
    for (int i = 0; i < NPIX_OUT; i++) {
        int d = abs((int)out_raw[i] - (int)exp_raw[i]);
        if (d > max_diff) max_diff = d;
        if (d > 1) {
            mismatches++;
            if (mismatches <= 10)
                printf("  mismatch pixel %d: kernel=%d expected=%d diff=%d\n",
                       i, out_raw[i], exp_raw[i], d);
        }
    }
    printf("Max diff: %d\n", max_diff);
    printf("Mismatches (diff>1): %d / %d\n", mismatches, NPIX_OUT);
    if (mismatches == 0) { printf("PASS\n"); return 0; }
    else { printf("FAIL\n"); return 1; }
}
