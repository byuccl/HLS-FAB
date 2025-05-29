#include <stdint.h>

#define SIGNAL_SIZE 1024
#define FILTER_ORDER 4

void iir_filter_float32(float *input_signal, float *output_signal, float *a_coeffs, float *b_coeffs) {
    float w[FILTER_ORDER + 1] = {0}; // Delay line for the filter

    for (int n = 0; n < SIGNAL_SIZE; n++) {
        // Shift the delay line
        for (int i = FILTER_ORDER; i > 0; i--) {
            w[i] = w[i - 1];
        }

        // Calculate the new output
        w[0] = input_signal[n];
        for (int i = 1; i <= FILTER_ORDER; i++) {
            w[0] -= a_coeffs[i] * w[i];
        }

        output_signal[n] = 0;
        for (int i = 0; i <= FILTER_ORDER; i++) {
            output_signal[n] += b_coeffs[i] * w[i];
        }
    }
}