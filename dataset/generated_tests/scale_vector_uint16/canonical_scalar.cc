#include <stdint.h>

#define VECTOR_SIZE 8192

void scale_vector_uint16(uint16_t *in_vector, uint16_t alpha, uint16_t *out_vector) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        out_vector[i] = in_vector[i] * alpha;
    }
}