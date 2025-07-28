import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 64

function_signature = "void dotproduct_bias_relu_bfloat16(ap_float<16,8>* in0, ap_float<16,8>* in1, ap_float<16,8>* out, ap_float<16,8> bias, uint32_t vector_size)"

description = f"""
This HLS kernel computes the dot product of two vectors of bfloat16 values, each of size {vector_size}. 
The result of the dot product is then added to a bfloat16 bias value. 
After adding the bias, a ReLU (Rectified Linear Unit) operation is applied, which sets any negative result to zero. 
The final result is a single bfloat16 value, which is written to the output.
Parameters:
    - First parameter: Pointer to the first input buffer of {vector_size} ap_float<16,8> values.
    - Second parameter: Pointer to the second input buffer of {vector_size} ap_float<16,8> values.
    - Third parameter: Pointer to the output buffer of a single ap_float<16,8> value.
    - Fourth parameter: A single ap_float<16,8> bias value.
    - Fifth parameter: An unsigned integer representing the size of the vectors, which is {vector_size}.
"""

inputs_info = [
    {
        "name": "in0",  # Name of the first input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "in1",  # Name of the second input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },
    {
        "name": "out",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (1,),  # Shape of the output buffer
        "total_elements": 1,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    },
    {
        "name": "bias",  # Name of the bias value
        "array_dimension": 0,  # Number of dimensions in the bias
        "shape": (),  # Shape of the bias
        "total_elements": 1,  # Total number of elements in the bias
        "dtype": "bfloat16",
        "iotype": "in",  # Input value for the kernel
        "min": -3,  # Minimum value for the bias
        "max": 3.0,  # Maximum value for the bias
    }
]

def behavioral(in0, in1, bias):
    s = bfloat16(np.dot(in0, in1)) + bias
    s_relu = max(s, bfloat16(0.0))
    return s_relu.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")