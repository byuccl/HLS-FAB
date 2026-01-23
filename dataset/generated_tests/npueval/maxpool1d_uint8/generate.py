import numpy as np
from hlsfab import PromptConstructor, generate_examples

num_values = 1024  # Length of the input vector
window_size = 2
stride = 2
output_length = (num_values - window_size) // stride + 1

function_signature = "void maxpool1d_uint8(ap_int<8> *in_buffer, ap_int<8> *out_buffer)"

short_description = f"A kernel that performs a 1D maxpool operation on an input vector with window_size and stride as runtime parameters."

description = f"""
An HLS kernel that performs a 1D max pooling operation on an input vector with window_size and stride as runtime parameters.
    First parameter: Pointer to the input buffer of {num_values} ap_int<8> values.
    Second parameter: Pointer to the output buffer of ap_int<8> values.
The kernel computes the maximum value of each window of size {window_size} in the input buffer,
with a stride of {stride}, and stores the result in the output buffer.
The output buffer will have a length of (input_length - window_size) // stride + 1.
The input buffer is expected to be at least {num_values} ap_int<8> values.
"""

inputs_info = [
    {
        "name": "in_buffer",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "uint8",
        "iotype": "in",  # Input buffer for the kernel
        "min": 0,  # Minimum value for the input buffer
        "max": 255,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "out_buffer",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (output_length,),  # Shape of the output buffer
        "total_elements": output_length,  # Total number of elements in the output buffer
        "dtype": "uint8",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    num_windows = (len(input_buffer) - window_size) // stride + 1
    output_buffer = np.zeros(num_windows, dtype=np.uint8)
    for i in range(num_windows):
        window = input_buffer[i*stride : i*stride + window_size]
        output_buffer[i] = np.max(window)
    return output_buffer

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")