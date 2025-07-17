import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

# Constants
vector_size = 256

function_signature = "void reciprocal_bfloat16(ap_float<16,8> *input_vector, ap_float<16,8> *output_vector)"

description = f"""
This HLS kernel computes the elementwise reciprocal (1/x) of a bfloat16 input vector of size {vector_size}, writing the result to an output buffer.
    First parameter: Pointer to the input buffer of {vector_size} ap_float<16,8> values.
    Second parameter: Pointer to the output buffer of {vector_size} ap_float<16,8> values.
The kernel processes each element of the input buffer, computes its reciprocal, and stores the result in the corresponding position in the output buffer.
"""

inputs_info = [
    {
        "name": "input_vector",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (vector_size,),  # Shape of the input buffer
        "total_elements": vector_size,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -10.0,  # Minimum value for the input buffer
        "max": 10.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output_vector",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (vector_size,),  # Shape of the output buffer
        "total_elements": vector_size,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_vector):
    # Prevent division by zero for behavioral model, though canonical has no check.
    result = np.empty_like(input_vector)
    result[:] = 1.0 / input_vector.astype(np.float32)
    result[input_vector == 0] = np.inf
    return result.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

tolerances = {"atol": 0.02, "rtol": 0.02}

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples,
    tolerances=tolerances
)

pc.write_json("kernel.json")