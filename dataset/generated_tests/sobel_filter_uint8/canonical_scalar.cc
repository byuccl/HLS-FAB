#include <stdint.h>

#define IMAGE_HEIGHT 128
#define IMAGE_WIDTH 128

void sobel_filter_uint8(uint8_t input_image[IMAGE_HEIGHT][IMAGE_WIDTH], int16_t output_image_x[IMAGE_HEIGHT][IMAGE_WIDTH], int16_t output_image_y[IMAGE_HEIGHT][IMAGE_WIDTH]) {
    // Sobel kernels
    int8_t sobel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int8_t sobel_y[3][3] = {
        {-1, -2, -1},
        {0,  0,  0},
        {1,  2,  1}
    };

    // Apply Sobel filter
    for (int i = 1; i < IMAGE_HEIGHT - 1; i++) {
        for (int j = 1; j < IMAGE_WIDTH - 1; j++) {
            int16_t gradient_x = 0;
            int16_t gradient_y = 0;

            // Convolve with Sobel kernels
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    gradient_x += input_image[i + ki][j + kj] * sobel_x[ki + 1][kj + 1];
                    gradient_y += input_image[i + ki][j + kj] * sobel_y[ki + 1][kj + 1];
                }
            }

            output_image_x[i][j] = gradient_x;
            output_image_y[i][j] = gradient_y;
        }
    }
}