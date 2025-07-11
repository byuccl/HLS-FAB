import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the image
HEIGHT = 128
WIDTH = 128

function_signature = """
void accumulate_squared_opencv(ap_uint<8><8> *in_gray, ap_uint<8><8> *in_gray1, ap_uint<16> *dst_hls)
// in_gray,   [HEIGHT][WIDTH] // 8-bit unsigned integer input image
// in_gray1,  [HEIGHT][WIDTH] // 8-bit unsigned integer input image
// dst_hls,   [HEIGHT][WIDTH] // 16-bit unsigned integer output image
"""

description = f"""
This HLS kernel implements a function to accumulate the square of an image to another image using OpenCV's accumulate square functionality.
Parameters to the function include two ap_uint<8><8> pointers and one ap_uint<16> pointer:
    The first parameter points to the start of the 2D input image array. The dimensions are {HEIGHT} x {WIDTH}. Each pixel is an 8-bit unsigned integer.
    The second parameter points to the start of another 2D input image array. The dimensions are {HEIGHT} x {WIDTH}. Each pixel is an 8-bit unsigned integer.
    The third parameter points to the start of the 2D output image array. The dimensions are {HEIGHT} x {WIDTH}. Each pixel is a 16-bit unsigned integer.
"""

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_gray = data["input_gray"]
        input_gray1 = data["input_gray1"]
        output_dst_hls = data["output_dst_hls"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_gray = np.random.randint(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8).flatten()
        input_gray1 = np.random.randint(0, 256, size=(HEIGHT, WIDTH), dtype=np.uint8).flatten()
        output_dst_hls = np.zeros(shape=(HEIGHT, WIDTH), dtype=np.uint16).flatten()
    inputs = [input_gray, input_gray1, output_dst_hls]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_gray, input_gray1):
    input_gray = input_gray.reshape((HEIGHT, WIDTH))
    input_gray1 = input_gray1.reshape((HEIGHT, WIDTH))
    squared = np.square(input_gray, dtype=np.uint16)
    return (squared + input_gray1.astype(np.uint16)).flatten()


examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")