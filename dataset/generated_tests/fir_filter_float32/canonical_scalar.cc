#include <stdint.h>

#define SIGNAL_SIZE 1024
#define FILTER_TAPS 64

void fir_filter_float32(float *signal, float *coefficients, float *output) {
    for (int i = 0; i < SIGNAL_SIZE; i++) {
        output[i] = 0;
        for (int j = 0; j < FILTER_TAPS; j++) {
            if (i - j >= 0) {
                output[i] += signal[i - j] * coefficients[j];
            }
        }
    }
}