import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
num_values = height * width

function_signature = """void bit_depth_convert_opencv(ap_uint<8> *input_img, int shift, ap_uint<16> *output_img)"""

description = f"""
This HLS kernel implements a function to convert the bit depth of an input image from 8-bit to 16-bit. 
Parameters to the function include two pointers and one integer:
    The first parameter is a pointer to the start of a {height * width}-element array of ap_uint<8> values representing the input image.
    The second parameter is an integer representing the shift value to be applied during conversion.
    The third parameter is a pointer to the start of a {height * width}-element array of ap_uint<16> values representing the output image.
"""

inputs_info = [
    {
        "name": "input_img",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "shift",  # Name of the input value
        "array_dimension": 0,  # Number of dimensions in the input vector
        "shape": (),  # Shape of the input vector
        "total_elements": 1,  # Total number of elements in the input vector
        "dtype": "int32",
        "iotype": "in",  # Input vector for the kernel
        "min": 1,  # Minimum value for the input vector
        "max": 2,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "output_img",  # Name of the result
        "array_dimension": 2,  # Number of dimensions in the result
        "shape": (height,width),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint16",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

def behavioral(input_img, shift):  
    # Instead of using numpy, let's work with raw Python types to avoid any format issues  
      
    # Validate shift parameter  
    if shift < 0 or shift > 8:  
        shift = 0  
      
    # Define the expected size (same as C++)  
    SIZE = 16384  
      
    # Create a Python list for the output (will be converted to numpy array later)  
    output = []  
      
    # Process each pixel exactly as in C++  
    for i in range(SIZE):  
        # Get the pixel value, default to 0 if out of bounds  
        pixel = input_img[i] if i < len(input_img) else 0  
          
        # Convert to 16-bit and shift, just like C++  
        # Use integer operations to ensure exact same behavior  
        converted_pixel = (int(pixel) & 0xFF) << shift  
          
        # Add to output list  
        output.append(converted_pixel)  
      
    # Convert to numpy array of proper type at the end  
    return np.array(output, dtype=np.uint16)   

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")