import numpy as np
import os
import torch    
from hlseval import PromptConstructor, generate_types, generate_examples

# Input and kernel dimensions    
input_height = 28    
input_width = 28    
kernel_size = 3    
stride = 1    

# Channel and batch configuration    
batch_size = 16    
in_channels = 3    
out_channels = 16  

# Calculate the dimensions of the output matrix with stride    
output_height = (input_height - kernel_size) / stride + 1      
output_width = (input_width - kernel_size) / stride + 1 

description = f"""
This HLS kernel implements a 2D convolution operation for neural networks. 
Parameters to the function include three 4D int16_t arrays, all intended to be interpreted as 4-dimensional arrays:
    The first parameter points to the start of the 4D input data array. The dimensions are [batch_size={batch_size}][in_channels={in_channels}][height={input_height}][width={input_width}].
    The second parameter points to the start of the the 4D kernel array. The dimensions are [out_channels={out_channels}][in_channels={in_channels}][kernel_size={kernel_size}][kernel_size={kernel_size}].
    The third parameter points to the start of the 4D ouput data array. The dimensions are [batch_size={batch_size}][in_channels={out_channels}][height={output_height}][width={output_width}].
The stride is fixed to be {stride}. All values are 16-bit integers."    
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):  
    if seed == -1: 
        data = np.load('data.npz')  
        input_matrix1 = data["input_matrix1"]
        input_matrix2 = data["input_matrix2"]
        output_matrix = data["output_matrix"]
    else:
        # Set a seed so the example inputs and outputs work correctly  
        np.random.seed(seed)  

        input_matrix1 = np.random.randint(-128, 128, size=(batch_size, in_channels, input_height, input_width), dtype=np.int16)  
        input_matrix2 = np.random.randint(-2, 2, size=(out_channels, in_channels, kernel_size, kernel_size), dtype=np.int16)  
        output_matrix = np.zeros(shape=(np.int16(batch_size), np.int16(out_channels), np.int16(output_height), np.int16(output_width)), dtype=np.int16)  
    inputs = [input_matrix1, input_matrix2, output_matrix]  

    types = generate_types(inputs)    
    return {"inputs": inputs, "types": types}    
  
# A function to generate a correct solution based on sample input matrices    
def behavioral(input_matrix1, input_matrix2):    
    # Convert numpy arrays to PyTorch tensors  
    input_tensor = torch.tensor(input_matrix1, dtype=torch.float32)  
    weight_tensor = torch.tensor(input_matrix2, dtype=torch.float32)  
      
    # Apply convolution using PyTorch's functional interface  
    output = torch.nn.functional.conv2d(input_tensor, weight_tensor, stride=stride)  
      
    # Convert back to int16 numpy array  
    return output.round().to(torch.int16).numpy()  

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")