#include <stdint.h>  

#define MATRIX_HEIGHT 32
#define MATRIX_WIDTH 24
#define MATRIX_DEPTH 64

void scale_matrix_3d_int8(  
    int8_t input[MATRIX_HEIGHT][MATRIX_WIDTH][MATRIX_DEPTH],    // 3D input matrix  
    int8_t scale_factor,                                        // Scalar value to multiply by  
    int8_t output[MATRIX_HEIGHT][MATRIX_WIDTH][MATRIX_DEPTH]    // 3D output matrix  
) {  
    // Iterate through each element of the 3D matrix  
    for (int i = 0; i < MATRIX_HEIGHT; i++) {  
        for (int j = 0; j < MATRIX_WIDTH; j++) {  
            for (int k = 0; k < MATRIX_DEPTH; k++) {  
                // Scale each element by the scale factor  
                output[i][j][k] = input[i][j][k] * scale_factor;  
            }  
        }  
    }  
}  