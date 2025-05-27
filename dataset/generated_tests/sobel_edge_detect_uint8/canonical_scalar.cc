#include <stdint.h>

#define IMAGE_HEIGHT 1024
#define IMAGE_WIDTH 1024

void sobel_edge_detect_uint8(uint8_t input_image[IMAGE_HEIGHT][IMAGE_WIDTH], uint8_t output_image[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    int8_t Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int8_t Gy[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1}
    };

    for (int i = 1; i < IMAGE_HEIGHT - 1; i++) {
        for (int j = 1; j < IMAGE_WIDTH - 1; j++) {
            int16_t sumX = 0;
            int16_t sumY = 0;

            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    sumX += input_image[i + ki][j + kj] * Gx[ki + 1][kj + 1];
                    sumY += input_image[i + ki][j + kj] * Gy[ki + 1][kj + 1];
                }
            }

            int16_t magnitude = abs(sumX) + abs(sumY);
            if (magnitude > 255) {
                magnitude = 255;
            }
            output_image[i][j] = (uint8_t)magnitude;
        }
    }
}