#include <stdint.h>

#define ARRAY_SIZE 65536

void mean_float32(float *in_array, float *mean) {
    float sum = 0.0;
    for (uint32_t i = 0; i < ARRAY_SIZE; i++) {
        sum += in_array[i];
    }
    *mean = sum / ARRAY_SIZE;
}