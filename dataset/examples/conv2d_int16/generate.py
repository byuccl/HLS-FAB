import numpy as np
import os
import torch    
from hlseval import PromptConstructor, generate_types, generate_examples

# Input and kernel dimensions    
input_height = 16    
input_width = 16    
kernel_size = 3    
stride = 1    

# Channel and batch configuration    
batch_size = 8    
in_channels = 3    
out_channels = 16  

# Calculate the dimensions of the output matrix with stride    
output_height = int((input_height - kernel_size) / stride + 1 )     
output_width = int((input_width - kernel_size) / stride + 1 )

function_signature = """
void conv2d_int16(ap_int<16> *input, ap_int<16> *kernel, ap_int<16> *output)
// input,        [BATCH_SIZE][IN_CHANNELS][INPUT_HEIGHT][INPUT_WIDTH]  
// kernel,       [OUT_CHANNELS][IN_CHANNELS][KERNEL_SIZE][KERNEL_SIZE]  
// output        [BATCH_SIZE][OUT_CHANNELS][OUTPUT_HEIGHT][OUTPUT_WIDTH]  
"""

description = f"""
This HLS kernel implements a 2D convolution operation for neural networks. 
Parameters to the function include three ap_int<16> arrays, all intended to be interpreted as 4-dimensional arrays:
    The first parameter points to the start of the 4D input data array. The dimensions are [batch_size={batch_size}][in_channels={in_channels}][height={input_height}][width={input_width}].
    The second parameter points to the start of the the 4D kernel array. The dimensions are [out_channels={out_channels}][in_channels={in_channels}][kernel_size={kernel_size}][kernel_size={kernel_size}].
    The third parameter points to the start of the 4D ouput data array. The dimensions are [batch_size={batch_size}][in_channels={out_channels}][height={output_height}][width={output_width}].
The stride is fixed to be {stride}. All values are 16-bit integers."    
"""

inputs_info = [
    {
        "name": "input",  # Name of the input matrix
        "array_dimension": 4,  # Number of dimensions in the input matrix
        "shape": (batch_size, in_channels, input_height, input_width),  # Shape of the input matrix
        "total_elements": batch_size * in_channels * input_height * input_width,  # Total number of elements in the input matrix
        "dtype": "int16",
        "iotype": "in",  # Input matrix for the kernel
        "min": -128,  # Minimum value for the input matrix
        "max": 127,  # Maximum value for the input matrix
    },  # Input matrix for the kernel
    {
        "name": "kernel",  # Name of the kernel matrix
        "array_dimension": 4,  # Number of dimensions in the kernel matrix
        "shape": (out_channels, in_channels, kernel_size, kernel_size),  # Shape of the kernel matrix
        "total_elements": out_channels * in_channels * kernel_size * kernel_size,  # Total number of elements in the kernel matrix
        "dtype": "int16",
        "iotype": "in",  # Kernel matrix for the kernel
        "min": -2,  # Minimum value for the kernel matrix
        "max": 2,  # Maximum value for the kernel matrix
    },  # Kernel matrix for the kernel
    {
        "name": "output",  # Name of the output matrix
        "array_dimension": 4,  # Number of dimensions in the output matrix
        "shape": (batch_size, out_channels, output_height, output_width),  # Shape of the output matrix
        "total_elements": batch_size * out_channels * output_height * output_width,  # Total number of elements in the output matrix
        "dtype": "int16",
        "iotype": "out",  # Output matrix for the kernel
    }  # Output matrix for the kernel
]

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
# def generate_inputs(seed=0):  
#     if seed == -1: 
#         data = np.load('data.npz')  
#         input_matrix1 = data["input_matrix1"]
#         input_matrix2 = data["input_matrix2"]
#         output_matrix = data["output_matrix"]
#     else:
#         # Set a seed so the example inputs and outputs work correctly  
#         np.random.seed(seed)  

#         input_matrix1 = np.random.randint(-128, 128, size=(batch_size, in_channels, input_height, input_width), dtype=np.int16).flatten()   
#         input_matrix2 = np.random.randint(-2, 2, size=(out_channels, in_channels, kernel_size, kernel_size), dtype=np.int16).flatten()  
#         output_matrix = np.zeros(shape=(np.int16(batch_size), np.int16(out_channels), np.int16(output_height), np.int16(output_width)), dtype=np.int16).flatten() 
#     inputs = [input_matrix1, input_matrix2, output_matrix]  

#     types = generate_types(inputs)    
#     return {"inputs": inputs, "types": types}    
  
# A function to generate a correct solution based on sample input matrices    
def behavioral(input_array_1d, weight_array_1d):  
    # Reshape 1D input array to 4D tensor [batch_size, in_channels, input_height, input_width]  
    input_tensor = torch.tensor(  
        input_array_1d,   
        dtype=torch.float32  
    ).reshape(batch_size, in_channels, input_height, input_width)  
      
    # Reshape 1D weight array to 4D tensor [out_channels, in_channels, kernel_size, kernel_size]  
    weight_tensor = torch.tensor(  
        weight_array_1d,   
        dtype=torch.float32  
    ).reshape(out_channels, in_channels, kernel_size, kernel_size)  
      
    # Apply convolution using PyTorch's functional interface  
    output = torch.nn.functional.conv2d(input_tensor, weight_tensor, stride=stride)  
      
    # Convert to int16 and flatten to 1D for output  
    # Output shape should be [batch_size, out_channels, output_height, output_width]  
    output_1d = output.round().to(torch.int16).flatten().numpy()  
      
    return output_1d  
    
examples = generate_examples(inputs_info)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")