#include <stdint.h>

#define HEIGHT 128
#define WIDTH 128

void bit_depth_convert_opencv(uint8_t *in_img, int shift, uint16_t *out_img) {
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            int index = i * WIDTH + j;
            out_img[index] = (uint16_t)(in_img[index]) << shift;
        }
    }
}