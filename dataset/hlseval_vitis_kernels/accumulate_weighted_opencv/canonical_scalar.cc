#include <stdint.h>

#define HEIGHT 128
#define WIDTH 128

void accumulate_weighted_opencv(uint8_t *src1, uint8_t *src2, uint16_t *dst, float alpha) {
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        dst[i] = (uint16_t)(alpha * src1[i] + (1 - alpha) * src2[i]);
    }
}