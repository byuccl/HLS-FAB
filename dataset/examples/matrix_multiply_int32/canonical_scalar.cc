#include <stdint.h>

#define MATRIX_1_HEIGHT 16
#define MATRIX_1_WIDTH 32
#define MATRIX_2_HEIGHT 32
#define MATRIX_2_WIDTH 24


void matrix_multiply_int32(
    int32_t matrix1[MATRIX_1_HEIGHT][MATRIX_1_WIDTH], 
    int32_t matrix2[MATRIX_2_HEIGHT][MATRIX_2_WIDTH], 
    int32_t result[MATRIX_1_HEIGHT][MATRIX_2_WIDTH])
{
// Ensure matrices are compatible for multiplication  
    if (MATRIX_1_WIDTH != MATRIX_2_HEIGHT) {  
        return; // Or handle dimension mismatch appropriately  
    }  
  
    for (int i = 0; i < MATRIX_1_HEIGHT; ++i) {  
        for (int j = 0; j < MATRIX_2_WIDTH; ++j) {  
            result[i][j] = 0; // Initialize result cell to zero  
            for (int k = 0; k < MATRIX_1_WIDTH; ++k) {  
                result[i][j] += matrix1[i][k] * matrix2[k][j];  
            }  
        }  
    } 
}