#include <stdint.h>

#define HEIGHT 128
#define WIDTH 128

void add_weighted_opencv(uint8_t *src1, uint8_t *src2, uint8_t *dst, float alpha, float beta, float gamma) {
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        float weighted_sum = alpha * src1[i] + beta * src2[i] + gamma;
        dst[i] = (uint8_t)(weighted_sum < 0 ? 0 : (weighted_sum > 255 ? 255 : weighted_sum));
    }
}