#include <stdint.h>
#define BUFFER_SIZE 32768

void mean_var_float64(double *in_buffer, double* out_mean, double* out_var) {
    double sum = 0;
    double sum_sq_diff = 0;
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        sum += in_buffer[i];
    }
    double temp_mean = sum / BUFFER_SIZE;
    for (int i = 0; i < BUFFER_SIZE; ++i) {
        sum_sq_diff += (in_buffer[i] - temp_mean) * (in_buffer[i] - temp_mean);
    }
    
    *out_mean = temp_mean;
    *out_var = sum_sq_diff / BUFFER_SIZE;

}