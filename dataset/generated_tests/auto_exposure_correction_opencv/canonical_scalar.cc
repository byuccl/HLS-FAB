#include <stdint.h>

#define HEIGHT 128
#define WIDTH 128
#define CHANNELS 3
#define HISTOGRAM_SIZE 256

void auto_exposure_correction_opencv(uint8_t *in_img, uint8_t *out_img) {
    // Convert RGB to HSV
    uint8_t hsv_img[HEIGHT * WIDTH * CHANNELS];
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        uint8_t r = in_img[i * CHANNELS];
        uint8_t g = in_img[i * CHANNELS + 1];
        uint8_t b = in_img[i * CHANNELS + 2];

        uint8_t max_val = r > g ? (r > b ? r : b) : (g > b ? g : b);
        uint8_t min_val = r < g ? (r < b ? r : b) : (g < b ? g : b);
        uint8_t delta = max_val - min_val;

        uint8_t v = max_val;
        uint8_t s = max_val == 0 ? 0 : (delta * 255) / max_val;
        uint8_t h = 0;

        if (delta != 0) {
            if (max_val == r) {
                h = 43 * (g - b) / delta;
            } else if (max_val == g) {
                h = 85 + 43 * (b - r) / delta;
            } else {
                h = 171 + 43 * (r - g) / delta;
            }
        }

        hsv_img[i * CHANNELS] = h;
        hsv_img[i * CHANNELS + 1] = s;
        hsv_img[i * CHANNELS + 2] = v;
    }

    // Histogram equalization on the V channel
    uint32_t histogram[HISTOGRAM_SIZE] = {0};
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        uint8_t v = hsv_img[i * CHANNELS + 2];
        histogram[v]++;
    }

    uint32_t cdf[HISTOGRAM_SIZE] = {0};
    cdf[0] = histogram[0];
    for (int i = 1; i < HISTOGRAM_SIZE; i++) {
        cdf[i] = cdf[i - 1] + histogram[i];
    }

    uint8_t lut[HISTOGRAM_SIZE];
    for (int i = 0; i < HISTOGRAM_SIZE; i++) {
        lut[i] = (uint8_t)((cdf[i] - cdf[0]) * 255 / (HEIGHT * WIDTH - cdf[0]));
    }

    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        hsv_img[i * CHANNELS + 2] = lut[hsv_img[i * CHANNELS + 2]];
    }

    // Convert HSV back to RGB
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        uint8_t h = hsv_img[i * CHANNELS];
        uint8_t s = hsv_img[i * CHANNELS + 1];
        uint8_t v = hsv_img[i * CHANNELS + 2];

        uint8_t region = h / 43;
        uint8_t remainder = (h - (region * 43)) * 6;

        uint8_t p = (v * (255 - s)) >> 8;
        uint8_t q = (v * (255 - ((s * remainder) >> 8))) >> 8;
        uint8_t t = (v * (255 - ((s * (255 - remainder)) >> 8))) >> 8;

        uint8_t r, g, b;
        switch (region) {
            case 0:
                r = v; g = t; b = p;
                break;
            case 1:
                r = q; g = v; b = p;
                break;
            case 2:
                r = p; g = v; b = t;
                break;
            case 3:
                r = p; g = q; b = v;
                break;
            case 4:
                r = t; g = p; b = v;
                break;
            default:
                r = v; g = p; b = q;
                break;
        }

        out_img[i * CHANNELS] = r;
        out_img[i * CHANNELS + 1] = g;
        out_img[i * CHANNELS + 2] = b;
    }
}