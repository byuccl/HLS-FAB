#define VECTOR_SIZE 262144

void relu_activation_float32(float *in_vector, float *out_vector) {
    for (int i = 0; i < VECTOR_SIZE; i++) {
        out_vector[i] = in_vector[i] > 0 ? in_vector[i] : 0;
    }
}

