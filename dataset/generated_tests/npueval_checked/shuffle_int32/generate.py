import numpy as np
from hlseval import PromptConstructor, generate_examples

# Constants
num_values = 256

function_signature = "void shuffle_int32(ap_int<32>* input, ap_int<32>* output, ap_int<32> seed)"

description = f"""
This HLS kernel performs an in-place Fisher–Yates shuffle on a vector of 256 32-bit integers. 
The function takes three parameters:
    1. A pointer to the input buffer containing {num_values} ap_int<32> values.
    2. A pointer to the output buffer where the shuffled result will be stored.
    3. A seed value of type ap_int<32> used to initialize the random number generator for shuffling.
The input buffer is copied to the output buffer before shuffling.
"""

inputs_info = [
    {
        "name": "input",  # Name of the input buffer
        "array_dimension": 1,  # Number of dimensions in the input buffer
        "shape": (num_values,),  # Shape of the input buffer
        "total_elements": num_values,  # Total number of elements in the input buffer
        "dtype": "int32",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1000,  # Minimum value for the input buffer
        "max": 1000,  # Maximum value for the input buffer
    },
    {
        "name": "output",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "int32",
        "iotype": "out",  # Output buffer for the kernel
    },
    {
        "name": "seed",  # Name of the seed parameter
        "array_dimension": 0,  # Scalar value
        "dtype": "int32",
        "iotype": "in",  # Input parameter for the kernel
        "min": 1,  # Minimum value for the seed
        "max": (2**32 - 1)//2,  # Maximum value for the seed
    }
]

def behavioral(input_buffer, seed):
    arr = input_buffer.copy()
    N = arr.shape[0]
    state = np.uint32(seed)
    for i in range(N - 1, 0, -1):
        state ^= (state << np.uint32(13))
        state ^= (state >> np.uint32(17))
        state ^= (state << np.uint32(5))
        j = int(state % np.uint32(i + 1))
        arr[i], arr[j] = arr[j], arr[i]
    return arr

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")