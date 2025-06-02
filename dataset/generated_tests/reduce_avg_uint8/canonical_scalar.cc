#include <stdint.h>

#define MATRIX_SIZE 128

void reduce_avg_uint8(uint8_t matrix[MATRIX_SIZE][MATRIX_SIZE], uint8_t result[MATRIX_SIZE]) {
    for (int j = 0; j < MATRIX_SIZE; j++) {
        uint32_t sum = 0;
        for (int i = 0; i < MATRIX_SIZE; i++) {
            sum += matrix[i][j];
        }
        result[j] = sum / MATRIX_SIZE;
    }
}