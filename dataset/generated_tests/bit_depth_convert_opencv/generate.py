import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128

function_signature = """void bit_depth_convert_opencv(ap_uint<8> *input_img, int shift, ap_uint<16> *output_img)"""
description = f"""
This HLS kernel implements a function to convert the bit depth of an input image from 8-bit to 16-bit. 
Parameters to the function include two pointers and one integer:
    The first parameter is a pointer to the start of a {height * width}-element array of ap_uint<8> values representing the input image.
    The second parameter is an integer representing the shift value to be applied during conversion.
    The third parameter is a pointer to the start of a {height * width}-element array of ap_uint<16> values representing the output image.
"""


def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_img = data["input_img"]
        input_shift = data["input_shift"]
        output_img = data["output_img"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_img = np.random.randint(0, 256, size=(height * width), dtype=np.uint8)
        input_shift = np.random.randint(0, 8, dtype=int)  # Shift value between 0 and 7
        output_img = np.zeros(height * width, dtype=np.uint16)
    inputs = [input_img, input_shift, output_img]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

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

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")