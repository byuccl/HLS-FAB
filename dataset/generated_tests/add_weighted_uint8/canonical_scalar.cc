#include <stdint.h>

#define IMAGE_HEIGHT 128
#define IMAGE_WIDTH 128

void add_weighted_uint8(
    uint8_t src1[IMAGE_HEIGHT][IMAGE_WIDTH],
    uint8_t src2[IMAGE_HEIGHT][IMAGE_WIDTH],
    float alpha,
    float beta,
    float gamma,
    uint8_t dst[IMAGE_HEIGHT][IMAGE_WIDTH]
) {
    for (int i = 0; i < IMAGE_HEIGHT; i++) {
        for (int j = 0; j < IMAGE_WIDTH; j++) {
            float weighted_sum = src1[i][j] * alpha + src2[i][j] * beta + gamma;
            if (weighted_sum > 255) {
                weighted_sum = 255;
            } else if (weighted_sum < 0) {
                weighted_sum = 0;
            }
            dst[i][j] = (uint8_t)weighted_sum;
        }
    }
}