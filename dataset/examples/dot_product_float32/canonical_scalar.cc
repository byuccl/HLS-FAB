#define VECTOR_SIZE 524288
void dot_product_float32(float *in_vector1, float *in_vector2, float *result)
{
    *result = 0;
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        *result += in_vector1[i] * in_vector2[i];
    }
}