#include <stdint.h>

#define VECTOR_SIZE 4096

void axpy_int32_res(int32_t alpha, int32_t *x_vector, int32_t *y_vector, int32_t *y_res) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        y_res[i] = alpha * x_vector[i] + y_vector[i];
    }
}