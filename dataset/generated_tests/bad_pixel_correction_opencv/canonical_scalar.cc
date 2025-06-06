#include <stdint.h>

#define HEIGHT 128
#define WIDTH 128

void bad_pixel_correction_opencv(uint8_t *img_in, uint8_t *img_out) {
    // Iterate over each pixel in the image
    for (int y = 0; y < HEIGHT; y++) {
        for (int x = 0; x < WIDTH; x++) {
            // Initialize min and max values with the current pixel value
            uint8_t min_val = img_in[y * WIDTH + x];
            uint8_t max_val = img_in[y * WIDTH + x];

            // Check the cross-diagonal neighbors
            int dx[] = {-2, -1, 0, 1, 2};
            int dy[] = {-2, -1, 0, 1, 2};

            for (int i = 0; i < 5; i++) {
                for (int j = 0; j < 5; j++) {
                    if ((i == 0 || i == 2 || i == 4) && (j == 0 || j == 2 || j == 4)) {
                        int nx = x + dx[j];
                        int ny = y + dy[i];
                        if (nx >= 0 && nx < WIDTH && ny >= 0 && ny < HEIGHT) {
                            uint8_t neighbor_val = img_in[ny * WIDTH + nx];
                            if (neighbor_val < min_val) {
                                min_val = neighbor_val;
                            }
                            if (neighbor_val > max_val) {
                                max_val = neighbor_val;
                            }
                        }
                    }
                }
            }

            // Correct the pixel value if it's outside the min-max range
            uint8_t pixel_val = img_in[y * WIDTH + x];
            if (pixel_val < min_val) {
                img_out[y * WIDTH + x] = min_val;
            } else if (pixel_val > max_val) {
                img_out[y * WIDTH + x] = max_val;
            } else {
                img_out[y * WIDTH + x] = pixel_val;
            }
        }
    }
}