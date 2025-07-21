# File to convert all of the npueval datasets to the new format
from openai import OpenAI
import os
import pathlib
from hlseval import HLSCoder, get_kernel_code
import json
import argparse
import logging

logger = logging.getLogger("CONVERSION")

SYSTEM_PROMPT = """
You are a part of a benchmarking system for HLS (High Level Synthesis).
You are very skilled both in writing python and C++ code.
Your job is to create the base code for tests within the benchmark.
Each test within the benchmark will have a Python file with a description of the task and functions to facilitate testing.
These files should follow the patterns set out by the provided examples.
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


    # Gather all the desired tests from the npueval dataset
    npueval_kernel_dir = "../npueval_private/dataset/kernels"
    npueval_kernels = []
    try: 
        kernels = os.listdir(npueval_kernel_dir)
        for kernel in kernels:
            npueval_kernels.append(kernel)
    except FileNotFoundError:
        logger.error(f"Directory {npueval_kernel_dir} not found. Please ensure it exists.")
        exit(1)




    # Open example files
    with open("dataset/npu_examples/abs_int8_old/generate.py", "r") as f:
        abs_generate_old = f.read()
    with open("dataset/npu_examples/abs_int8_new/generate.py", "r") as f:
        abs_generate_new = f.read()
    with open("dataset/npu_examples/abs_int8_old/canonical_scalar.cc", "r") as f:
        abs_canonical_old = f.read()

    with open("dataset/npu_examples/avgpool1d_bfloat16_old/generate.py", "r") as f:
        avgpool_generate_old = f.read()
    with open("dataset/npu_examples/avgpool1d_bfloat16_new/generate.py", "r") as f:
        avgpool_generate_new = f.read()
    with open("dataset/npu_examples/avgpool1d_bfloat16_old/canonical_scalar.cc", "r") as f:
        avgpool_canonical_old = f.read()


    # # Generate 'generate.py' scripts
    prompt = f"""
I am generating a series of tests to assess the performance of HLS C++ code. Each test consists of a file titled 'generate.py'.
The 'generate.py' scipt contains one python function 'behavioral' and other variables that help the test creation and process. These other variables are 'function_signature', 'description', and 'inputs_info'
\t'function_signature' is a string that defines the function signature of the kernel being tested. This line can be derived from the 'canonical_scalar.cc' that will be provided.
\t\t Also for the datatypes in the function signature, use ap_int types from the hls library 'ap_int.h' and ap_float types from 'ap_float.h' when there is a bfloat16.
\t 'description' is a string that describes the kernel being tested, including its inputs and outputs. The old 'generate.py' has a description but make it more detailed as you see in the examples.
\t'inputs_info' is a dictionary that describes the inputs to the kernel, including
\t\t their names, types, shapes, range of values and io type (input, output, or inout).
\t'behavioral' performs the operation described by the kernel in python. This function generally uses external libraries such as numpy or pytorch to ensure accuracy even with complex kernels.
\t\t The old 'generate.py' already has this, so it is only necessary to copy it over.
\tAdditionally, the generate script includes a description of the kernel, defines several values, and calls a 'PromptConstructor' class. This class pulls from values passed in to create a description of the prompt complete with examples and dataflow information.
The following is an example of the old 'generate.py' file for the 'abs_int8' test:\n```\n{abs_generate_old}\n```\n
The following is an example of the old 'canonical_scalar.cc' file for the 'abs_int8' test:\n```\n{abs_canonical_old}\n```\n
The following is an example of the new 'generate.py' file for the 'abs_int8' test:\n```\n{abs_generate_new}\n```\n
The following is an example of the old 'generate.py' file for the 'avgpool1d_bfloat16' test:\n```\n{avgpool_generate_old}\n```\n
The following is an example of the new 'generate.py' file for the 'avgpool1d_bfloat16' test:\n```\n{avgpool_generate_new}\n```\n
The following is an example of the old 'canonical_scalar.cc' file for the 'avgpool1d_bfloat16' test:\n```\n{avgpool_canonical_old}\n```\n
    """

   
        
    model = "gpt-4o"
    temperature = 0.0
    top_p = 1.0
    attempts = 1



    #Chaging the list so that it doesnt go through all 100
    # npueval_kernels = [
    #     # "argmax_int32",
    #     "cast_bfloat16_to_int8",
    #     # "conv2d_int32",
    # ]   



    for i, desired_test in enumerate(npueval_kernels):

        coder = HLSCoder(
            model=model,
            hls_flow="vitis",
            temperature=temperature,
            workflow=workflow,
            top_p=top_p,
            url=url,
            attempts=attempts,
            system_prompt=SYSTEM_PROMPT,
            rag=False
        )

        # if i < 47:
        #     logger.info("Make the kernels after 10 tests for demonstration purposes.")
        #     continue
        # if i >= 52:
        #     logger.info("Only make the first 20 kernels for demonstration purposes.")
        #     break
        pathlib.Path(f"dataset/generated_tests/npueval/{desired_test}").mkdir(parents=True, exist_ok=True)

        # Get the kernel code
        with open(f"{npueval_kernel_dir}/{desired_test}/generate.py", "r") as f:
            desired_test_generate = f.read()

        with open(f"{npueval_kernel_dir}/{desired_test}/canonical_scalar.cc", "r") as f:
            desired_test_canonical = f.read()

        # Check if the generate.py file already exists
        if os.path.exists(f"dataset/generated_tests/npueval/{desired_test}/generate.py"):
            logger.info(
                f"The file dataset/generated_tests/npueval/{desired_test}/generate.py already exists. Skipping file."
            )
            continue
        
        generate_line = f"""
Your task is to create a 'generate.py' script for the following test:\n{desired_test}
Use the same naming convention for variables using input_ for inputs and output_ for outputs and inout_ for combinations of the two.
Produce only the function, no main, no additional examples or code. Don't even wrap it in a code block.
If there are constants then include them at the top of the file as variables.\n
The following is the old 'generate.py' file for the {desired_test} test:\n```\n{desired_test_generate}\n```\n
The following is the 'canonical_scalar.cc' file for the {desired_test} test:\n```\n{desired_test_canonical}\n```\n

Please use these old files as a reference to create the new 'generate.py' file for the {desired_test} test.
        """
        logger.info(f"Generating code for {desired_test}")
        # logger.debug(f"Prompt for {desired_test}: {prompt + generate_line}")
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
                f"dataset/generated_tests/npueval/{desired_test}/generate.py", "x"
            ) as file:
                file.write(response["response"])
        except FileExistsError:
            logger.info(
                f"The file dataset/generated_tests/npueval/{desired_test}/generate.py already exists. Skipping file."
            )
