import numpy as np
import os
from hlsfab import PromptConstructor, generate_types, generate_examples

# Image dimensions
height = 128
width = 128
filter_size = 3
num_values = width * height

function_signature = """void bilateral_filter_opencv(ap_uint<8> *input_image, double sigma_color, double sigma_space, ap_uint<8> *output_image)"""

description = f"""
This HLS kernel implements a bilateral filter on a 128x128 image. The bilateral filter smoothens the image while preserving edges by considering both spatial and intensity differences. The window size for this kernel is 3.
Parameters to the function include two ap_uint<8> pointers and two double values:
    The first pointer points to the start of a {height * width}-element array of ap_uint<8> values representing the input image.
    The second parameter is a double value representing the sigma for color.
    The third parameter is a double value representing the sigma for space.
    The fourth pointer points to the start of a {height * width}-element array of ap_uint<8> values representing the output image.
"""

inputs_info = [
    {
        "name": "input_image",  # Name of the input vector
        "array_dimension": 2,  # Number of dimensions in the input vector
        "shape": (height,width),  # Shape of the input vector
        "total_elements": num_values,  # Total number of elements in the input vector
        "dtype": "uint8",
        "iotype": "in",  # Input vector for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 127,  # Maximum value for the input vector
    },  # Input vector for the kernel
    {
        "name": "sigma_color",  # Name of the result
        "array_dimension": 0,  # Number of dimensions in the result
        "shape": (),  # Shape of the result
        "total_elements": 1,  # Total number of elements in the result
        "dtype": "float64",
        "iotype": "in",  # Input for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 1,  # Maximum value for the input vector
    },  
    {
        "name": "sigma_space",  # Name of the result
        "array_dimension": 0,  # Number of dimensions in the result
        "shape": (),  # Shape of the result
        "total_elements": 1,  # Total number of elements in the result
        "dtype": "float64",
        "iotype": "in",  # Input for the kernel
        "min": 0,  # Minimum value for the input vector
        "max": 1,  # Maximum value for the input vector
    },  
    {
        "name": "output_image",  # Name of the result
        "array_dimension": 2,  # Number of dimensions in the result
        "shape": (height,width),  # Shape of the result
        "total_elements": num_values,  # Total number of elements in the result
        "dtype": "uint8",
        "iotype": "out",  # Output for the kernel
    }  # Output for the kernel
]

def behavioral(input_image, sigma_color, sigma_space):
    input_image_2d = input_image.reshape((height, width))
    output_image_2d = np.zeros_like(input_image_2d)
    half_filter = filter_size // 2
    color_coeff = -0.5 / (sigma_color * sigma_color)
    space_coeff = -0.5 / (sigma_space * sigma_space)

    for y in range(height):
        for x in range(width):
            sum_weights = 0.0
            sum_intensity = 0.0
            center_pixel = input_image_2d[y, x]

            for fy in range(-half_filter, half_filter + 1):
                for fx in range(-half_filter, half_filter + 1):
                    ny, nx = y + fy, x + fx
                    if 0 <= ny < height and 0 <= nx < width:
                        neighbor_pixel = input_image_2d[ny, nx]
                        color_diff = center_pixel - neighbor_pixel
                        space_diff = fy * fy + fx * fx

                        weight = np.exp(space_diff * space_coeff) * np.exp(color_diff * color_diff * color_coeff)
                        sum_intensity += weight * neighbor_pixel
                        sum_weights += weight

            output_image_2d[y, x] = np.clip(sum_intensity / sum_weights, 0, 255)

    return output_image_2d.flatten()

examples = generate_examples(inputs_info=inputs_info)

pc = PromptConstructor(
    function_signature=function_signature,
    description=description,
    behavioral=behavioral,
    examples=examples
)

pc.write_json("kernel.json")