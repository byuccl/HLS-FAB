#include <stdint.h>      
  
// Input and kernel dimensions  
#define INPUT_HEIGHT 28  
#define INPUT_WIDTH 28  
#define KERNEL_SIZE 3  
#define STRIDE 1  
  
// Channel and batch configuration  
#define BATCH_SIZE 16  
#define IN_CHANNELS 3  
#define OUT_CHANNELS 16  

// Calculate the dimensions of the output matrix with stride    
#define OUTPUT_HEIGHT ((INPUT_HEIGHT - KERNEL_SIZE) / STRIDE + 1)      
#define OUTPUT_WIDTH ((INPUT_WIDTH - KERNEL_SIZE) / STRIDE + 1) 
  
void conv2d_int16(  
    int16_t input[BATCH_SIZE][IN_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH],       // [BATCH_SIZE][IN_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH]  
    int16_t kernel[OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE],      // [OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE]  
    int16_t output[BATCH_SIZE][OUT_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH]       // [BATCH_SIZE][OUT_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH]  
) {      
    
      
    // Process each sample in the batch  
    for (int b = 0; b < BATCH_SIZE; ++b) {  
        // Process each output channel  
        for (int oc = 0; oc < OUT_CHANNELS; ++oc) {  
            // Initialize output cells to zero  
            for (int i = 0; i < OUTPUT_HEIGHT; ++i) {      
                for (int j = 0; j < OUTPUT_WIDTH; ++j) {      
                    output[b][oc][i][j] = 0;     
                }  
            }  
              
            // Sum contributions from all input channels  
            for (int ic = 0; ic < IN_CHANNELS; ++ic) {  
                // Perform the convolution operation    
                for (int i = 0; i < OUTPUT_HEIGHT; ++i) {      
                    for (int j = 0; j < OUTPUT_WIDTH; ++j) {      
                        for (int ki = 0; ki < KERNEL_SIZE; ++ki) {      
                            for (int kj = 0; kj < KERNEL_SIZE; ++kj) {      
                                // Sum across channels  
                                output[b][oc][i][j] +=   
                                    input[b][ic][i * STRIDE + ki][j * STRIDE + kj] *   
                                    kernel[oc][ic][ki][kj];      
                            }      
                        }      
                    }      
                }  
            }  
        }  
    }      
}  
