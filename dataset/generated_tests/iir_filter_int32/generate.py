import numpy as np
import os
from hlseval import PromptConstructor, generate_types, generate_examples

# Set the signal length and filter order
signal_length = 1024
filter_order = 4

function_signature = """void iir_filter_int32(ap_int<32> *input_signal, ap_int<32> *output_signal, ap_int<32> *a_coeffs, ap_int<32> *b_coeffs)"""

description = f"""
This HLS kernel implements an infinite impulse response (IIR) filter applied to a 32-bit integer signal.
Parameters to the function include four ap_int<32> pointers:
    The first parameter points to the start of an array of {signal_length} int32 values. This is the input signal.
    The second parameter points to the start of an array of {signal_length} int32 values. This is the output signal.
    The third parameter points to an array of {filter_order + 1} int32 values. These are the 'a' coefficients of the filter.
    The fourth parameter points to an array of {filter_order + 1} int32 values. These are the 'b' coefficients of the filter.
"""

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

        input_signal = np.random.randint(-1000, 1000, size=(signal_length), dtype=np.int32)
        output_signal = np.zeros(signal_length, dtype=np.int32)
        input_a_coeffs = np.random.randint(-10, 10, size=(filter_order + 1), dtype=np.int32)
        input_b_coeffs = np.random.randint(-10, 10, size=(filter_order + 1), dtype=np.int32)
    inputs = [input_signal, output_signal, input_a_coeffs, input_b_coeffs]

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

def behavioral(input_signal, a_coeffs, b_coeffs):
    y = np.zeros_like(input_signal, dtype=np.int32)
    for n in range(signal_length):
        y[n] = 0
        for i in range(filter_order + 1):
            if n - i >= 0:
                y[n] += b_coeffs[i] * input_signal[n - i]
        for j in range(1, filter_order + 1):
            if n - j >= 0:
                y[n] -= a_coeffs[j] * y[n - j]
    return y

examples = generate_examples(generate_inputs)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")