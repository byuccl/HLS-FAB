#define VECTOR_SIZE 65536
void affine_float32(float *in_vector, float in_scale, float in_bias, float *out_vector)
{
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        out_vector[i] = in_vector[i] * in_scale + in_bias;
    }
}

