import numpy as np
from ml_dtypes import bfloat16
from hlseval import PromptConstructor, generate_examples

num_values = 256  # Length of the input vector

function_signature = "void variance_bfloat16(ap_float<16,8> *input, ap_float<16,8> *output)"

short_description = (
    "This kernel computes the variance of a bfloat16 input vector (length 256) using the definition "
    "Var(X) = E[X^2] - (E[X])^2, and writes the scalar (bfloat16) result to the output buffer (length 1)."
)

description = f"""
This HLS kernel computes the variance of a bfloat16 input vector of length {num_values} using the definition:
    Var(X) = E[X^2] - (E[X])^2
The result is a scalar bfloat16 value written to the output buffer.
Parameters:
    - First parameter: Pointer to the input buffer of {num_values} ap_float<16,8> values.
    - Second parameter: Pointer to the output buffer of a single ap_float<16,8> value.
The kernel calculates the mean and mean of squares of the input values to determine the variance.
If the computed variance is negative due to round-off errors, it is clamped to zero.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -3,  # Minimum value for the input buffer
        "max": 3.0,  # Maximum value for the input buffer
    },  # Input buffer for the kernel
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 0,  # Number of dimensions in the output buffer
        "shape": (),  # Shape of the output buffer
        "total_elements": 1,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }  # Output buffer for the kernel
]

def behavioral(input_buffer):
    vals = input_buffer.astype(np.float32)
    mean = np.mean(vals)
    mean_sq = np.mean(vals ** 2)
    var = mean_sq - mean ** 2
    if var < 0:
        var = 0.0
    return np.array([bfloat16(var)])

examples = generate_examples(inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")