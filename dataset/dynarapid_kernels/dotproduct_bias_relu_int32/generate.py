import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void dotproduct_bias_relu_int32(ap_int<32>* in0, ap_int<32>* in1, ap_int<32>* out, ap_int<32> bias)"

description = f"""
This HLS kernel computes the dot product of two vectors of 32-bit integer numbers, each of length {num_values}. 
The result is then added to a scalar 32-bit integer bias. 
After adding the bias, a ReLU (Rectified Linear Unit) operation is applied, which sets any negative result to zero. 
Finally, the result is clamped to fit within the range of an 32-bit integer and written to a single-element output buffer.
Parameters to the function include:
    - Two input pointers, each pointing to the start of a {num_values}-element array of ap_int<32> values.
    - A scalar ap_int<32> bias value.
    - An output pointer to a single ap_int<32> value where the result will be stored.
"""

inputs_info = [
    {
        "name": "in0",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1024,  # Minimum value for the input buffer
        "max": 1024,  # Maximum value for the input buffer
    },
    {
        "name": "in1",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1024,  # Minimum value for the input buffer
        "max": 1024,  # Maximum value for the input buffer
    },
    {
        "name": "out",  # Name of the output buffer
        "array_dimension": 0,  # Scalar output
        "shape": (),  # Shape of the output
        "total_elements": 1,  # Total number of elements in the output
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    },
    {
        "name": "bias",  # Name of the bias input
        "array_dimension": 0,  # Scalar input
        "shape": (),  # Shape of the input
        "total_elements": 1,  # Total number of elements in the input
        "dtype": "int32",
        "iotype": "in",  # Input for the kernel
        "min": -2048,  # Minimum value for the bias
        "max": 2048,  # Maximum value for the bias
    }
]

def behavioral(in0, in1, bias):
    acc = np.dot(in0.astype(np.int32), in1.astype(np.int32)) + int(bias)
    acc = max(acc, 0)
    acc = np.clip(acc, -2**31, 2**31-1)
    return np.int32(acc)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")