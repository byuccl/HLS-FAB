import numpy as np
from ml_dtypes import bfloat16
from hlsfab import PromptConstructor, generate_examples

num_values = 256  # Length of the input vectors

function_signature = "void hingeloss_bfloat16(ap_float<16,8> *pred, ap_float<16,8> *label, ap_float<16,8> *out)"

description = f"""
An HLS kernel that computes the element-wise hinge loss for two bfloat16 arrays of length {num_values}.
The function takes three parameters:
    1. Pointer to the prediction buffer of {num_values} ap_float<16,8> values.
    2. Pointer to the label buffer of {num_values} ap_float<16,8> values.
    3. Pointer to the output buffer of {num_values} ap_float<16,8> values.
For each index i, the output is calculated as: output[i] = max(0, 1 - label[i] * pred[i]),
where label is typically +1 or -1.
"""

inputs_info = [
    {
        "name": "pred",  # Name of the prediction buffer
        "array_dimension": 1,  # Number of dimensions in the prediction buffer
        "shape": (num_values,),  # Shape of the prediction buffer
        "total_elements": num_values,  # Total number of elements in the prediction buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -2.0,  # Minimum value for the prediction buffer
        "max": 2.0,  # Maximum value for the prediction buffer
    },
    {
        "name": "label",  # Name of the label buffer
        "array_dimension": 1,  # Number of dimensions in the label buffer
        "shape": (num_values,),  # Shape of the label buffer
        "total_elements": num_values,  # Total number of elements in the label buffer
        "dtype": "bfloat16",
        "iotype": "in",  # Input buffer for the kernel
        "min": -1.0,  # Minimum value for the label buffer
        "max": 1.0,  # Maximum value for the label buffer
    },
    {
        "name": "out",  # Name of the output buffer
        "array_dimension": 1,  # Number of dimensions in the output buffer
        "shape": (num_values,),  # Shape of the output buffer
        "total_elements": num_values,  # Total number of elements in the output buffer
        "dtype": "bfloat16",
        "iotype": "out",  # Output buffer for the kernel
    }
]

def behavioral(pred, label):
    t = 1.0 - label * pred
    h = np.maximum(0.0, t)
    return h.astype(bfloat16)

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")