#include <stdint.h>

#define M 64
#define N 64
#define K 64

void mat_mult_int16(
    int16_t matrix1[M][N], 
    int16_t matrix2[N][K], 
    int16_t result[M][K])
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < K; ++j) {
            result[i][j] = 0; // Initialize result cell to zero
            for (int k = 0; k < N; ++k) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}