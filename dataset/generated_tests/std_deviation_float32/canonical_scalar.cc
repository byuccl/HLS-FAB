#include <stdint.h>
#include <cmath>
#define BUFFER_SIZE 131072

void std_deviation_float32(float *in_buffer, float *out_std_dev) {
    float sum = 0;
    float sum_sq_diff = 0;
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        sum += in_buffer[i];
    }
    float mean = sum / BUFFER_SIZE;
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        sum_sq_diff += (in_buffer[i] - mean) * (in_buffer[i] - mean);
    }
    *out_std_dev = std::sqrt(sum_sq_diff / BUFFER_SIZE);
}