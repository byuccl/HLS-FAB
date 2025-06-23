from openai import OpenAI
import pathlib
from hlseval import HLSCoder, get_kernel_code
import json
import argparse
import logging

logger = logging.getLogger("__main__")

SYSTEM_PROMPT = """
You are a part of a benchmarking system for HLS (High Level Synthesis).
You are very skilled both in writing python and C++ code.
Your job is to create the base code for tests within the benchmark.
Each test should have a C++ implementation that hasn't been accellerated at all. This will serve as a functional example and provide a method to test HLS solutions.
While this C++ implementation will not be synthesized for HLS, you should avoid using techniques and libraries that would fail to synthesize.
Additionally, each test within the benchmark will have a Python file with a description of the task and functions to facilitate testing.
Both these files should follow the patterns set out by the provided examples.
Produce only the code requested, no main, no additional examples or code, and no wrappers of any kind.
"""
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="A program to tests for the HLSEval benchmark")

    parser.add_argument("--log", dest='loglevel', help="desired loglevel", default="WARNING")

    byu_gpt_url = "https://api.openai.com/v1/chat/completions"
    amd_gpt_url = "https://llm-api.amd.com/openai/deployments/swe-gpt4o-exp1/chat/completions"

    parser.add_argument('workflow', choices=['amd', 'byu'],  
                        help="Which workflow to follow for API calls ('amd' or 'byu')")  
    args = parser.parse_args()  
    workflow = args.workflow
    if workflow == "byu":
        url = byu_gpt_url
    elif workflow == "amd":
        url = amd_gpt_url
    else:
        logger.error("Invalid workflow selection")

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    # Set up prompt

    desired_test = "accumulate_squared_opencv"
    desired_test_description = (
        """
    The accumulate square function adds the square of an image (src1) to the accumulator image (src2) and generates the accumulated result (dst).

    dst[i] = src1[i]^2 + src2[i]

    dst[i] is the i-th element of the output array.
    src1[i] is the i-th element of the first input array.
    src2[i] is the i-th element of the second input array.


    The input image will be 128x128 pixels
    Each pixel of the output image is represented by a 1-channel, 8-bit, unsigned integer
    Each pixel of the output image is represented by a 1-channel, 16-bit, unsigned integer
    The image will be passed in and out as single-row arrays. Defined constants for 'HEIGHT' and 'WIDTH' (both set to 128) should be used to correctly break the array into a 2D image.
    Implementation details should match the OpenCV Reduce implementation unless otherwise specified

    void accumulate_squared_opencv(uint8_t*)in_gray.data, (uint8_t*)in_gray1.data, (uint16_t*)dst_hls.data);
        """
    )      

    # Open example files
    with open("dataset/examples/absolute_value_int8/generate.py", "r") as f:
        abs_generate = f.read()
    with open("dataset/examples/affine_float32/generate.py", "r") as f:
        affine_generate = f.read()
    with open("dataset/examples/conv2d_int16/generate.py", "r") as f:
        conv2d_generate = f.read()
    with open("dataset/examples/dot_product_float32/generate.py", "r") as f:
        dot_generate = f.read()
    with open("dataset/examples/matrix_multiply_int32/generate.py", "r") as f:
        mat_mul_generate = f.read()
    with open("dataset/examples/mean_var_float64/generate.py", "r") as f:
        mean_var_generate = f.read()
    with open("dataset/examples/relu_activation_float32/generate.py", "r") as f:
        relu_generate = f.read()
    with open("dataset/examples/scale_matrix_3d_int8/generate.py", "r") as f:
        scale_mat_generate = f.read()
    with open("dataset/examples/subtract_int64/generate.py", "r") as f:
        subtract_generate = f.read()
    with open("dataset/examples/zero_mean_norm_float64/generate.py", "r") as f:
        zero_mean_generate = f.read()

    # Generate 'canonical_scalar.cc'
    prompt = f"""
    I am generating a series of tests to assess the performance of HLS C++ code. Each test consists of files titled 'cononical_scalar.cc' and 'generate.py'.\n\n
    The canonical scalar function contains code to perform the necessary function in C++.\n
    The generate scipt contains two python functions: 'generate_inputs' and 'behavioral'. \n
    \t'generate_inputs' generates numpy arrays or scalar values for each parameter in the 'canonical_scalar.cc' file. Input arrays are randomly filled with values well within the minimums and maximums of the data type. This function also calls the external 'generate_types' function to tell the caller what type of data each parameter is.\n
    \t'behavioral' performs the operation described by the kernel in python. This function generally uses external libraries such as numpy or pytorch to ensure accuracy even with complex kernels.\n
    \tAdditionally, the generate script includes a description of the kernel, defines several values, and calls a 'PromptConstructor' class. This class pulls from values passed in to create a description of the prompt complete with examples and dataflow information.\n
    \n
    The following is an example of the matching 'generate.py' file for the same 'absolute_value_int8' test:\n```\n{abs_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'affine_float32' test:\n```\n{affine_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'conv2d_int16' test:\n```\n{conv2d_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'dot_product_float32' test:\n```\n{dot_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'matrix_multiply_int32' test:\n```\n{mat_mul_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'mean_var_float64' test:\n```\n{mean_var_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'relu_activation_float32' test:\n```\n{relu_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'scale_matrix_3d_int8' test:\n```\n{scale_mat_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'subtract_int64' test:\n```\n{subtract_generate}\n```\n\n
    The following is an example of the matching 'generate.py' file for the same 'zero_mean_norm_float64' test:\n```\n{zero_mean_generate}\n```\n\n
    """

   
        
    model = "gpt-4o"
    temperature = 0.0
    top_p = 1.0
    attempts = 1
    coder = HLSCoder(
        model=model,
        temperature=temperature,
        workflow=workflow,
        top_p=top_p,
        url=url,
        attempts=attempts,
        system_prompt=SYSTEM_PROMPT
    )

    pathlib.Path(f"dataset/generated_tests/{desired_test}").mkdir(parents=True, exist_ok=True)


    generate_line = f"""
    Your task is to create a 'generate.py script for the following test:\n{desired_test}: {desired_test_description}'\n\n
    Use the same naming convention for variables using input_ for inputs and output_ for outputs and inout_ for combinations of the two.
    Produce only the function, no main, no additional examples or code. Don't even wrap it in a code block.
    If there are constants then include them at the top of the file as variables. 
    Please ensure that the behavioral function performs a correct implementation of the kernel in Python, using numpy or pytorch as needed.
    """
    response = coder(prompt + generate_line)

    result = {
        "code": coder.extract_codeblock(response["response"]),
        "stats": {
            "token_usage": response["token_usage"],
            "history": response["history"],
        },
    }

    # Write just the code to a file
    try:
        with open(
            f"dataset/generated_tests/{desired_test}/generate.py", "x"
        ) as file:
            file.write(response["response"])
    except FileExistsError:
        logger.info(
            f"The file dataset/generated_tests/{desired_test}/generate.py already exists. Skipping file."
        )
