#define VECTOR_SIZE 65536

void zero_mean_norm_float64(double *in_vector, double *out_vector, double *mean)
{
    double sum;
    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        sum += in_vector[i];
    }
    *mean = sum / VECTOR_SIZE;

    for (int i = 0; i < VECTOR_SIZE; i++)
    {
        out_vector[i] = in_vector[i] - *mean;
    }
}

