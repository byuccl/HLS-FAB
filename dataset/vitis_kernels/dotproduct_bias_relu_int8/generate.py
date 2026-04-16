import numpy as np
from hlsfab import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void dotproduct_bias_relu_int8(ap_int<8>* in0, ap_int<8>* in1, ap_int<8>* out, ap_int<8> bias)"

description = f"""
This HLS kernel computes the dot product of two vectors of 8-bit integer numbers, each of length {num_values}. 
The result is then added to a scalar 8-bit integer bias. 
After adding the bias, a ReLU (Rectified Linear Unit) operation is applied, which sets any negative result to zero. 
Finally, the result is clamped to fit within the range of an 8-bit integer and written to a single-element output buffer.
Parameters to the function include:
    - Two input pointers, each pointing to the start of a {num_values}-element array of ap_int<8> values.
    - A scalar ap_int<8> bias value.
    - An output pointer to a single ap_int<8> value where the result will be stored.
"""

inputs_info = [
    {
        "name": "in0",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -20,  # Minimum value for the input buffer
        "max": 20,  # Maximum value for the input buffer
    },
    {
        "name": "in1",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int8",
        "iotype": "in",  # Input buffer for the kernel
        "min": -20,  # Minimum value for the input buffer
        "max": 20,  # Maximum value for the input buffer
    },
    {
        "name": "out",  # Name of the output buffer
        "array_dimension": 0,  # Scalar output
        "shape": (),  # Shape of the output
        "total_elements": 1,  # Total number of elements in the output
        "dtype": "int8",
        "iotype": "out",  # Output buffer for the kernel
    },
    {
        "name": "bias",  # Name of the bias input
        "array_dimension": 0,  # Scalar input
        "shape": (),  # Shape of the input
        "total_elements": 1,  # Total number of elements in the input
        "dtype": "int8",
        "iotype": "in",  # Input for the kernel
        "min": -10,  # Minimum value for the bias
        "max": 10,  # Maximum value for the bias
    }
]

def behavioral(in0, in1, bias):
    acc = np.dot(in0.astype(np.int32), in1.astype(np.int32)) + int(bias)
    acc = max(acc, 0)
    acc = np.clip(acc, -128, 127)
    return np.int8(acc)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")