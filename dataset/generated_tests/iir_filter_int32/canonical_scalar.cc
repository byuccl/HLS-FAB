#include <stdint.h>

#define SIGNAL_LENGTH 1024
#define FILTER_ORDER 4

void iir_filter_int32(int32_t *input_signal, int32_t *output_signal, int32_t *a_coeffs, int32_t *b_coeffs) {
    int32_t y[SIGNAL_LENGTH] = {0};

    for (int n = 0; n < SIGNAL_LENGTH; n++) {
        y[n] = 0;
        for (int i = 0; i <= FILTER_ORDER; i++) {
            if (n - i >= 0) {
                y[n] += b_coeffs[i] * input_signal[n - i];
            }
        }
        for (int j = 1; j <= FILTER_ORDER; j++) {
            if (n - j >= 0) {
                y[n] -= a_coeffs[j] * y[n - j];
            }
        }
        output_signal[n] = y[n];
    }
}