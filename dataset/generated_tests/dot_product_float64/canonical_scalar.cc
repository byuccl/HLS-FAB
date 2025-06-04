#define VECTOR_SIZE 1024

void dot_product_float64(double *in_vector1, double *in_vector2, double *result) {
    *result = 0;
    for (int i = 0; i < VECTOR_SIZE; i++) {
        *result += in_vector1[i] * in_vector2[i];
    }
}