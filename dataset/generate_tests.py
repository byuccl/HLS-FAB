from openai import OpenAI
import pathlib
from hlsfab import HLSCoder, get_kernel_code
import json
import argparse
import logging

logger = logging.getLogger("__main__")

SYSTEM_PROMPT = """
You are a part of a benchmarking system for HLS (High Level Synthesis).
You are very skilled both in writing python and C++ code.
Your job is to create the base code for tests within the benchmark.
Each test within the benchmark will have a Python file with a description of the task and functions to facilitate testing.
This files should follow the patterns set out by the provided examples.
Produce only the code requested, no main, no additional examples or code, and no wrappers of any kind.
"""
if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="A program to tests for the HLSFAB benchmark")
    kernel_name = input("Name of Kernel: ")
    user_description = input("Description: ")

    print(kernel_name)
    print(user_description)

    parser.add_argument("--log", dest='loglevel', help="desired loglevel", default="WARNING")

    url = "https://openrouter.ai/api/v1/chat/completions"


    # parser.add_argument('workflow', choices=['amd', 'byu'],  
    #                     help="Which workflow to follow for API calls ('amd' or 'byu')")  
    args = parser.parse_args()  
    # workflow = args.workflow
    # if workflow == "byu":
    #     url = url
    # elif workflow == "amd":
    #     url = amd_gpt_url
    # else:
    #     logger.error("Invalid workflow selection")

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)

    # Open example files
    with open("dataset/examples/abs_int8/generate.py", "r") as f:
        abs_generate = f.read()
    with open("dataset/examples/conv2d_int32/generate.py", "r") as f:
        conv2d_generate = f.read()
    with open("dataset/examples/dotproduct_bfloat16/generate.py", "r") as f:
        dot_generate = f.read()
    with open("dataset/examples/elementwise_min_float32/generate.py", "r") as f:
        min_generate = f.read()
    with open("dataset/examples/variance_bfloat16/generate.py", "r") as f:
        var_generate = f.read()
    with open("dataset/examples/relu_int8/generate.py", "r") as f:
        relu_generate = f.read()
    with open("dataset/examples/scale_matrix_3d_int8/generate.py", "r") as f:
        scale_mat_generate = f.read()
    with open("dataset/examples/vectorsubtract_int64/generate.py", "r") as f:
        subtract_generate = f.read()


    # Generate 'generate.py' scripts
    prompt = f"""
    I am generating a series of tests to assess the performance of HLS C++ code. Each test consists of a file titled 'generate.py'.\n\n
    The 'generate.py' scipt contains a 'behavioral' function which performs the function described in the description as well as a 'function_signature' which sets up a C++ function prototype for the function. \n
    \tAdditionally, the generate script includes a description of the kernel, defines several values, and calls a 'PromptConstructor' class. This class pulls from values passed in to create a description of the prompt complete with examples and dataflow information.\n
    \n
    The following is an example of the 'generate.py' file for the 'abs_int8' test:\n```\n{abs_generate}\n```\n\n
    The following is an example of the 'generate.py' file for the 'conv2d_int32' test:\n```\n{conv2d_generate}\n```\n\n
    The following is an example of the 'generate.py' file for the 'dotproduct_bfloat16' test:\n```\n{dot_generate}\n```\n\n
    The following is an example of the 'generate.py' file for the 'elementwise_min_float32' test:\n```\n{min_generate}\n```\n\n
    The following is an example of the 'generate.py' file for the 'variance_bfloat16' test:\n```\n{var_generate}\n```\n\n
    The following is an example of the 'generate.py' file for the 'relu_int8' test:\n```\n{relu_generate}\n```\n\n
    The following is an example of the 'generate.py' file for the 'scale_matrix_3d_int8' test:\n```\n{scale_mat_generate}\n```\n\n
    The following is an example of the 'generate.py' file for the 'vectorsubtract_int64' test:\n```\n{subtract_generate}\n```\n\n
    """

   
        
    model = "openai/gpt-4o"
    temperature = 0.0
    top_p = 1.0
    attempts = 1
    coder = HLSCoder(
        model=model,
        hls_flow="vitis",
        temperature=temperature,
        workflow='openrouter',
        top_p=top_p,
        url=url,
        attempts=attempts,
        system_prompt=SYSTEM_PROMPT
    )


    try:
        pathlib.Path(f"dataset/generated_tests/{kernel_name}").mkdir(parents=True, exist_ok=False)
        generate_line = f"""
        Your task is to create a 'generate.py script for the following test:\n{kernel_name}: {user_description}'\n\n
        Use the same naming convention for variables using input_ for inputs and output_ for outputs and inout_ for combinations of the two.
        Produce only the function, no main, no additional examples or code. Don't even wrap it in a code block.
        If there are constants then include them at the top of the file as variables. 
        Please ensure that the behavioral function performs a correct implementation of the kernel in Python, using numpy or pytorch as needed.
        The examples will give you an example of how to take the description and transform it into a working generate.py file.
        """
        logger.debug(f"Generating code for {kernel_name} with prompt: {prompt + generate_line}")
        response = coder(prompt + generate_line)
        logger.debug(f"Response for {kernel_name}: {response}")

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
                f"dataset/generated_tests/{kernel_name}/generate.py", "x"
            ) as file:
                file.write(response["response"])
        except FileExistsError:
            logger.info(
                f"The file dataset/generated_tests/{kernel_name}/generate.py already exists. Skipping file."
            )
    except FileExistsError:
        logger.warning("The desired test already exists check the 'generated_tests' folder before continuing.")



