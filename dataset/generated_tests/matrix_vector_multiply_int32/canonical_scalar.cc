#include <stdint.h>

#define MATRIX_HEIGHT 64
#define MATRIX_WIDTH 128

void matrix_vector_multiply_int32(
    int32_t matrix[MATRIX_HEIGHT][MATRIX_WIDTH], 
    int32_t vector[MATRIX_WIDTH], 
    int32_t result[MATRIX_HEIGHT])
{
    for (int i = 0; i < MATRIX_HEIGHT; ++i) {
        result[i] = 0; // Initialize result cell to zero
        for (int j = 0; j < MATRIX_WIDTH; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}