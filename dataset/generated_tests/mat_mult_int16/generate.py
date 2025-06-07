import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the dimensions of the matrices
M = 64
N = 64
K = 64

description = f"""
This HLS kernel implements a function to multiply two matrices of 16-bit integers.
Parameters to the function include three int16_t 2D arrays, all intended to be interpreted as 2-dimensional arrays:
    The first parameter points to the start of the first 2D input data array. The dimensions are {M} x {N}.
    The second parameter points to the start of the second 2D input data array. The dimensions are {N} x {K}.
    The third parameter points to the start of the 2D output data array. The dimensions are {M} x {K}.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1:
        data = np.load('data.npz')
        input_matrix1 = data["input_matrix1"]
        input_matrix2 = data["input_matrix2"]
        output_matrix = data["output_matrix"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_matrix1 = np.random.randint(-32768, 32767, size=(M, N), dtype=np.int16)
        input_matrix2 = np.random.randint(-32768, 32767, size=(N, K), dtype=np.int16)
        output_matrix = np.zeros((M, K), dtype=np.int16)
    inputs = [input_matrix1, input_matrix2, output_matrix]

    types = generate_types(inputs)
    return {"inputs": inputs, "types": types}

def behavioral(input_matrix1, input_matrix2):
    return np.matmul(input_matrix1, input_matrix2).astype(np.int16)

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")