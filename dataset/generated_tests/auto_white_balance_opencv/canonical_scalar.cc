#include <stdint.h>
#include <algorithm>

#define HEIGHT 2160
#define WIDTH 3840
#define CHANNELS 3

void auto_white_balance_opencv(uint8_t *in_img, uint8_t *out_img, float thresh, float inputMin, float inputMax, float outputMin, float outputMax) {
    float sum[CHANNELS] = {0.0f, 0.0f, 0.0f};
    int count[CHANNELS] = {0, 0, 0};

    // Calculate the sum of each channel based on the saturation threshold
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        uint8_t R = in_img[i * CHANNELS];
        uint8_t G = in_img[i * CHANNELS + 1];
        uint8_t B = in_img[i * CHANNELS + 2];

        float max_val = std::max({R, G, B});
        float min_val = std::min({R, G, B});
        float saturation = (max_val - min_val) / max_val;

        if (saturation < thresh) {
            sum[0] += R;
            sum[1] += G;
            sum[2] += B;
            count[0]++;
            count[1]++;
            count[2]++;
        }
    }

    // Calculate the average for each channel
    float avg[CHANNELS];
    for (int c = 0; c < CHANNELS; c++) {
        avg[c] = count[c] > 0 ? sum[c] / count[c] : 0.0f;
    }

    // Normalize each channel
    for (int i = 0; i < HEIGHT * WIDTH; i++) {
        for (int c = 0; c < CHANNELS; c++) {
            float value = in_img[i * CHANNELS + c];
            float normalized = (value - avg[c]) * (outputMax - outputMin) / (inputMax - inputMin) + outputMin;
            out_img[i * CHANNELS + c] = static_cast<uint8_t>(std::clamp(normalized, 0.0f, 255.0f));
        }
    }
}