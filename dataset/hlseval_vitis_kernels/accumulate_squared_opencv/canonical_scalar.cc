#include <stdint.h>

#define HEIGHT 128
#define WIDTH 128

void accumulate_squared_opencv(uint8_t *src1, uint8_t *src2, uint16_t *dst) {
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        uint16_t squared = (uint16_t)src1[i] * (uint16_t)src1[i];
        dst[i] = squared + (uint16_t)src2[i];
    }
}