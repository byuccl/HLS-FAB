<!--
Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.

Author: Preston Walker, AMD
SPDX-License-Identifier: MIT
-->

# HLSEval
## Overview of files and directories
### dataset
- **`dataset_tests.txt`**
  - This file contains a list of several potential tests for hlseval with a short description. These tests can be added to `generate_tests.py` to create a `canonical_scalar.cc` and a `generate.py` file.
- **`generate_hlseval.py`**
  - Goes through each test in the `kernels` folder, looks for a `kernel.json` file, and comples them into a single `hlseval.json` file. Also generates `kernel_list.txt`. Each of these files is used to run the kernel testbenches.
- **`generate_kernels.sh`**
  - Goes through each test in the `kernels` folder, looks for a `generate.py` file, and runs it. Each `generate.py` file should create a file called `kernel.json` with all the json data needed to run a test.
- **`generate_tests.py`**
  - This test uses the OpenAI API key to generate `canonical_scalar.cc` and `generate.py` files for a test described at the top of the file.
- **`hlseval.json`**
  - This file should have a line for each kernel to be run. When calling `generate_hlseval.py`, the `hlseval.json` file is not cleared.
#### examples
- This folder should contain the 10 examples tests (`absolute_value_int8`, `affine_float32`, `conv2d_int16`, `dot_product_float32`, `matrix_multiply_int32`, `mean_var_float64`, `relu_activation_float32`, `scale_matrix_3d_int8`, `subtract_int64`, and `zero_mean_norm_float64`). These are used by `generate_tests.py` as examples of how to create a `canonical_scalar.cc` and `generate.py` file.
#### generated_tests
- This folder is where the output of `generate_tests.py` is sent. It can act as a bank of kernels so you don't have to run them all or delete them each time you run hlseval
#### kernels
- This folder should contain each kernel you intend to run hlseval on. By clearing the existing `hsleval.json` file, running `generate_kernels.sh`, and then `generate_hlseval.py`, the `hlseval.json` file will contain a line for each test in this folder. From there you are ready to run hlseval
### hlseval
- This is a package of separate helpful modules for the rest of the project.
- **`csim.py`**
  - This module contains functions to generate a C++ testbench file using the behavioral function from the kernel's `generate.py` script and run the kernel/testbench using the Vitis C Simulation tool. 
- **`datset.py`**
  - This python module contains the class PromptConstructor which is used by each `generate.py` in order to create multishot prompts with examples for each kernel. 
- **`hlscoder.py`**
  - This python module contains the class HLSCoder which is the interface between the code and the LLM API calls. It is used for each completion as well as each test generation. This file also contains functions to interface with the RAG database and use it in code generation.
- **`parser.py`**
  - This python module takes the reports and logs from the Vitis tools output and parses through them gathering all relevant data. The output is put into the `<kernel_name>_data.json` files.
- **`run_cpp.py`** *depreciated*
  - This python module has helper functions for `run_tests.py` in the `testing` directory. It takes the inputs and outputs information and runs it in C to test the correctness of the function. 
- **`run_tests.py`** *depreciated*
  - This python module is used after the LLMs have created their completions and is used to check the correctness of the code generated. It goes through every kernel and model that has been generated and then checks to see if the behavioral from `generate.py` and the function in the `<kernel_name>.cpp` have the same Outputs. The results that are generated are outputted into `test_results.json`.
- **`utils.py`**
  - This python module holds many helper functions that are used by other scripts. 
- **`vitis.py`**
  - This python module contains a function to run Vitis C Synthesys using a subprocess.
### results
#### solutions
- This folder is not contained in the repository but all output from the project will go here. It will have subfolders named after each LLM Model you are running (i.e.`gpt-4o`). These will be divided by the number of attempts and then by kernel names. Each kernel folder will contain both the hls C code and the Vitis output. 
- **`<kernel_name>.cpp`** 
  - This is the outputted C++ file that the model generates. 
- **`<kernel_name>_data.json`**
  - This json file will contain a dictionary of results from the synthesis report. It contains data like latency, as well as BRAM, LUT and FF utilization. It also contains any warnings or errors from the logs if there are any. 
### other files
- **`dockerfile`**
  - Quickly sets up a docker container with all python packages and requirements for HLSEval. The docker container does not contain Vitis tools. Any calls to the Vitis tools must be commented out to run in this way. *This has not been updated for some time. It likely will not intsall all necessary python packages*
- **`hls_completions.py`** *depreciated*
  - This file contains the `main` function. This function will go through every prompt in `dataset/hlseval.json` file and generate completions for each model and output them to the `results` folder. The function will then run HLS synthesis with the Vitis tools.
  - **`hls_completions_loop.py`**
  - This file contains the `main` function. This function will go through every prompt in `dataset/hlseval.json` file and generate completions for each model and output them to the `results` folder. The function will then run C Simulation and C Synthesis with the Vitis tools. These results will be used in the next iteration of code generation to create a feedback loop
- **`requirements.txt`**
  - Contains a list of the required Python packages. This should facilitate setting up an environment without using the dockerfile.
## Instructions for Running Tests with HLSEval
>python venv environment
1. Generate the tests: Run the `generate_tests.py` script from the `dataset` directory. 
  - You can specify which test you want generated by changing the variables `desired_test` and `desired_test_description` at the top of the file.
  - The output should be the `canonical_scalar.cc` and `generate.py` files in the directory `generated_tests/<desired_test>`
2. Add the Test Kernels: Copy which generated tests that you want to analyze into the `kernels` folder. This is where the kernels will be generated.
3. Create the kernels: Run the `generate_kernels.sh` script.
  - The output should be `kernel.json` in the `kernels` folder.
4. Generate the HLSEval: Run the script `generate_hlseval.py`.
  - The output should be `hlseval.jsonl`and `kernel_list.txt` inside the `dataset` directory.
5. Generating completions: Run the `hls_completions_loop.py` in the `hlseval` main directory.
  - It is required to add some argmuments on the command line. Based on which API key you are using, there may be a different poll request URL.
    - Currently we only support two the public polling url and the AMD azure url. To specify either put byu for public or amd for AMD's workflow. 
  - There are other flags that can be added on the command line that add specific actions of the script. 
    * **Mode** The `--mode` flag gives you an option between `optimization`(default) and `correction` mode. Correction mode will only loop back if the first attempt does not compile in vitis tools. 
    * **Goals:** The `--goals` flag is used to specify which goals you would like to achieve with optimization mode. Simply write strings in double quotes stating what you would optimize. i.e. `"Reduce latency" "minimize warnings"`
  - Example: ```python hls_completions_loop.py amd --mode optimization --goals "Reduce BRAM Usage"```
  - Don't forget to activate Vitis e.g. `source tools/Xilinx/Vitis/2024.2/settings64.sh`
  - The output will be in a `solutions` folder in the `results` directory.
6. Analyze results: Look at the results from the scripts.
  - Results are stored in the `results/solutions/<model>/<kernel_name>` folder. A separate folder for each attempt will be present. The file `<kernel_name>.cpp` contains the generated HLS code, and the file `<kernel_name>_data.json` contains results from the C Simulation and C Synthesis. The original output logs should all have been saved as well. 
7. Makefile Completions: There is a Makefile rule for replacing the individual steps 3-6.
  - `make` from within the `hlseval` directory will run steps 3-5.
  - `make clean` will clean the unnecessary files for analysis.

## Building and running with Docker *depreciated*
> Currently only the python environment has been containerized. To run hlseval with Vitis tools, instead of running the docker container, you will need to install Vitis on your computer and set up a Python virtual environment.
### Build the docker container using the dockerfile
Run the command:\
`docker build -t hlseval .` 
### Run the container
Run the following command, making sure to replace `$OPENAI_API_KEY` with a valid api key to run OpenAI models. If this environment variable is already set on your computer, the command will work as is.\
`docker run --rm --name HLSEval -e OPENAI_API_KEY=$OPENAI_API_KEY hlseval`