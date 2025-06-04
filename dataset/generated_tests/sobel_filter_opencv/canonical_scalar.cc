#include <stdint.h>

#define ROWS 128
#define COLS 128

void sobel_filter_opencv(uint8_t* img_inp, uint8_t* img_out1, uint8_t* img_out2) {
    // Sobel kernels
    int8_t Gx[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1}
    };

    int8_t Gy[3][3] = {
        {-1, -2, -1},
        {0,  0,  0},
        {1,  2,  1}
    };

    // Iterate over each pixel in the image
    for (int i = 0; i < ROWS; i++) {
        for (int j = 0; j < COLS; j++) {
            // Initialize gradient values
            int16_t grad_x = 0;
            int16_t grad_y = 0;

            // Apply Sobel filter only if not on the border
            if (i > 0 && i < ROWS - 1 && j > 0 && j < COLS - 1) {
                for (int ki = -1; ki <= 1; ki++) {
                    for (int kj = -1; kj <= 1; kj++) {
                        uint8_t pixel = img_inp[(i + ki) * COLS + (j + kj)];
                        grad_x += pixel * Gx[ki + 1][kj + 1];
                        grad_y += pixel * Gy[ki + 1][kj + 1];
                    }
                }
            }

            // Set the output gradients
            img_out1[i * COLS + j] = (uint8_t)(grad_x > 255 ? 255 : (grad_x < 0 ? 0 : grad_x));
            img_out2[i * COLS + j] = (uint8_t)(grad_y > 255 ? 255 : (grad_y < 0 ? 0 : grad_y));
        }
    }
}