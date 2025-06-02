import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the number of values in the signal and filter order
signal_size = 1024
filter_order = 4

description = f"""
This HLS kernel implements an Infinite Impulse Response (IIR) filter applied to a 32-bit floating point signal.
Parameters to the function include four float32 pointers:
    The first pointer points to an array containing {signal_size} float values. This is the input signal.
    The second pointer points to an array containing {signal_size} float values. This is the output signal.
    The third pointer points to an array containing {filter_order + 1} float values. These are the 'a' coefficients of the filter.
    The fourth pointer points to an array containing {filter_order + 1} float values. These are the 'b' coefficients of the filter.
"""
source_path = os.path.join(os.path.dirname(__file__), "canonical_scalar.cc")

def generate_inputs(seed=0):
    if seed == -1: 
        data = np.load('data.npz')  
        input_signal = data["input_signal"]
        output_signal = data["output_signal"]
        input_a_coeffs = data["input_a_coeffs"]
        input_b_coeffs = data["input_b_coeffs"]
    else:
        # Set a seed so the example inputs and outputs work correctly
        np.random.seed(seed)

        input_signal = np.random.rand(signal_size).astype(np.float32) * 2 - 1
        output_signal = np.zeros(signal_size, dtype=np.float32)
        input_a_coeffs = np.random.rand(filter_order + 1).astype(np.float32) * 2 - 1
        input_b_coeffs = np.random.rand(filter_order + 1).astype(np.float32) * 2 - 1

    inputs = [input_signal, output_signal, input_a_coeffs, input_b_coeffs]
    types = generate_types(inputs)  

    return {"inputs": inputs, "types": types}  

def behavioral(input_signal, a_coeffs, b_coeffs):
    w = np.zeros(filter_order + 1, dtype=np.float32)
    output_signal = np.zeros_like(input_signal)

    for n in range(signal_size):
        w[1:] = w[:-1]
        w[0] = input_signal[n] - np.dot(a_coeffs[1:], w[1:])
        output_signal[n] = np.dot(b_coeffs, w)

    return output_signal

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    source_path=source_path,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")