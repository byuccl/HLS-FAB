#include <stdint.h>

#define HEIGHT 128
#define WIDTH 128

void reduce_avg_opencv(uint8_t *in_img, int dimension, uint8_t *out_img) {
    if (dimension == 0) {
        // Reduce to a single row
        for (int j = 0; j < WIDTH; j++) {
            int sum = 0;
            for (int i = 0; i < HEIGHT; i++) {
                sum += in_img[i * WIDTH + j];
            }
            out_img[j] = sum / HEIGHT;
        }
    } else if (dimension == 1) {
        // Reduce to a single column
        for (int i = 0; i < HEIGHT; i++) {
            int sum = 0;
            for (int j = 0; j < WIDTH; j++) {
                sum += in_img[i * WIDTH + j];
            }
            out_img[i] = sum / WIDTH;
        }
    }
}