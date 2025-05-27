#include <stdint.h>
#define BUFFER_SIZE 131072

void subtract_int64(int64_t *in_buffer1, int64_t *in_buffer2, int64_t* out_buffer) {
    for (uint32_t i = 0; i < BUFFER_SIZE; i++) {
        out_buffer[i] = in_buffer1[i] - in_buffer2[i];
    }
}