import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Set the number of values in the signal and filter taps
signal_size = 1024
filter_taps = 64

function_signature = """void fir_filter_float32(float *input_signal, float *filter_coefficients, float *output_signal)"""

description = f"""
This HLS kernel implements a finite impulse response (FIR) filter on a 32-bit floating-point signal.
Parameters to the function include three float32 pointers:
    The first pointer points to an array containing {signal_size} float values. This is the input signal.
    The second pointer points to an array containing {filter_taps} float values. These are the filter coefficients.
    The third pointer points to an array containing {signal_size} float values. This is the output signal.
"""

# A function to return the inputs and types to the function. 
# Inputs are determined by the parameters to the kernel found in the 'canonical_scalar.cc' file. 
# Types are determined by the 'generate_types' function.
def generate_inputs(seed=0):  
    if seed == -1: 
        data = np.load('data.npz')  
        input_signal = data["input_signal"]
        input_coefficients = data["input_coefficients"]
        output_signal = data["output_signal"]
    else:
        # Set a seed so the example inputs and outputs work correctly  
        np.random.seed(seed)  

        input_signal = np.random.rand(signal_size).astype(np.float32) * 2 - 1  # Random values between -1 and 1
        input_coefficients = np.random.rand(filter_taps).astype(np.float32) * 2 - 1  # Random values between -1 and 1
        output_signal = np.zeros(signal_size, dtype=np.float32)  
    inputs = [input_signal, input_coefficients, output_signal]  

    types = generate_types(inputs)  
    return {"inputs": inputs, "types": types}  

# A function to generate a correct solution based on sample input vectors
def behavioral(input_signal, input_coefficients):
    output_signal = np.zeros_like(input_signal)
    for i in range(len(input_signal)):
        for j in range(len(input_coefficients)):
            if i - j >= 0:
                output_signal[i] += input_signal[i - j] * input_coefficients[j]
    return output_signal

examples = generate_examples(generate_inputs)

# Create the prompt constructor.
pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")