#include <stdint.h>
#define BUFFER_SIZE 16384

void absolute_value_int8(int8_t *in_buffer, int8_t* out_buffer) {
    for (uint32_t i = 0; i < BUFFER_SIZE; i++) {
        int8_t value = in_buffer[i];
        if (value < 0) {
            out_buffer[i] = -value;
        } else {
            out_buffer[i] = value;
        }
    }
}