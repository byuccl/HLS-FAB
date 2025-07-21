import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 256  # Length of the input vector
window_size = 4
stride = 4
output_length = num_values // stride

function_signature = "void avgpool1d_relu_bfloat16(ap_float<16,8> *in_buffer, ap_float<16,8> *out_buffer)"

description = f"""
This HLS kernel performs a 1D average pooling operation followed by a ReLU activation on an input vector of bfloat16 values.
Parameters to the function include two ap_float<16,8> pointers:
    The first pointer points to the start of a {num_values}-element array of ap_float<16,8> values to be used as input data.
    The second pointer points to the start of a {output_length}-element array of ap_float<16,8> values to be modified and viewed as the output to the function.
The kernel computes the average of each window of size {window_size} in the input buffer, with a stride of {stride}, applies the ReLU activation function, and stores the result in the output buffer.
The output buffer will have a length of {output_length}.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (output_length,),  # Shape of the output buffer
        "total_elements": output_length,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):

    output_buffer = np.zeros(output_length, dtype=bfloat16)
    for i in range(output_length):
        window = input_buffer[i * stride : i * stride + window_size].astype(np.float32)
        avg = window.mean()
        relu_avg = np.maximum(avg, 0.0)
        output_buffer[i] = relu_avg.astype(bfloat16)
    return output_buffer

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")