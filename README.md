<!--
Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.

Author: Preston Walker, AMD
SPDX-License-Identifier: MIT
-->

# HLS-FAB
## Getting Started
### Prerequisites
- Have docker installed (used when running DynaRapid)
- Have python 3.10 or newer
- Have Vitis installed and sourced (We use version 2024.2)
- Set api_key environment variable:
`export OPENROUTER_API_KEY=<Add API Key>`

### Clone and prepare the repo 
```
git clone git@github.com:Preston-Walker/HLS-FAB.git
cd HLS-FAB
make setup
source venv/bin/activate
```


### Prep for run
```
cd dataset
# Add just one kernel for the test run
cp -r examples/abs_int8 kernels
cd ..
make dataset
```
To set which LLM models to use, edit the `dataset/llm_models.txt` file.

#### Set up docker container for dynarapid
```
wget https://gist.githubusercontent.com/clavin-xlnx/7ff0b7238aa5f5de4ff2dac7392d5132/raw/a130d502374cfa2779e59f918a2248e61fec5363/Dockerfile.dynarapid
mv Dockerfile.dynarapid Dockerfile
docker build -t my-llvm6-dynarapid-image .
docker run -v <Path to HLS-FAB reposotory>/HLS-FAB:/mount --name dynarapid_container -t my-llvm6-dynarapid-image
```
or simply
```
make setup_docker
```
Exit the docker container

### Run Vitis Completions
Sample of run command:\
`make run_vitis_completions LOG=DEBUG N=5 K=1`\
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `N` option: Allows you to determine the number of generation attempts per kernel. Setting this to `1` would make an LLM call for each kernel without incorporating feedback. Setting `N` to a value of `2` or more will incorporate a feedback loop where the results from each attempt are passed back to the LLM to improve upon the result each time.
- `K` option: Allows you to determine the number of independent runs for each kernel. Setting this to `1` would run the framework 1 time with all the `N` attempts and then finish. `2` or more `K` Would start the process again after without the need to run another command.
<!-- - `RAG` option: *Currently Not Implemented* -->


### Run DynaRapid Completions
Sample of run command:\
`make run_dynarapid_completions LOG=DEBUG N=4 K=2 RAPIDWRIGHT=TRUE`\
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `N` option: Allows you to determine the number of generation attempts per kernel. Setting this to `1` would make an LLM call for each kernel without incorporating feedback. Setting `N` to a value of `2` or more will incorporate a feedback loop where the results from each attempt are passed back to the LLM to improve upon the result each time.
- `K` option: Allows you to determine the number of independent runs for each kernel. Setting this to `1` would run the framework 1 time with all the `N` attempts and then finish. `2` or more `K` Would start the process again after without the need to run another command.
- `RAPIDWRIGHT` option: If omitted, or set to `FALSE`, Dynamatic will run without RapidWright, resulting in a `.dot` file, but no placed and routed circut. Adding the `DYNAMATIC` option and setting it to anything other than `FALSE` will run both Dynamatic and RapidWright. The value `TRUE` is used here for clarity.


### Results
Results for all runs will be found in the `results/solutions` folder and will be organized as follows:

```
results/
├── solutions/
│   ├── vitis/              (or dynarapid)
│   │   ├── rag/            (or no_rag)
|   |   |   ├── openai/     (or other company name)
|   │   │   │   ├── gtp-5/  (or other model name)
|   │   │   │   │   ├── 1/      (pass@k iteration number)
|   │   │   │   │   │   ├── kernel_name_1/
|   │   │   │   │   │   │   ├── attempts_1/
|   │   │   │   │   │   │   │   ├── kernel_name_1.cpp
|   │   │   │   │   │   │   │   ├── kernel_name_1.h
|   │   │   │   │   │   │   │   ├── kernel_name_1_tb.cpp
|   │   │   │   │   │   │   │   └── ...synthesis files...
|   │   │   │   │   │   │   ├── attempts_2/
|   │   │   │   │   │   │   └── ...
|   │   │   │   │   │   ├── kernel_name_2/
|   │   │   │   │   │   └── ...
|   │   │   │   │   ├── logs/
|   │   │   │   │   │   └── timing.log
|   │   │   │   │   ├── 2/
|   │   │   │   │   └── ...
|   │   │   │   └── ...
|   |   |   └── ...
│   │   └── no_rag/
│   │       └── ...
│   └── dynarapid/
│       └── ...
└── logs/
    └── completions_YYYY-MM-DD_HH:MM:SS.log
```

**Folder Structure Explanation:**
- `vitis/` or `dynarapid/`: HLS flow tool used
- `rag/` or `no_rag/`: Whether RAG (Retrieval-Augmented Generation) was enabled
- `{model_name}/`: LLM model used for generation (e.g., gpt-4o, claude-3)
- `{iter}/`: Pass@k iteration number (1, 2, 3, etc.)
- `{kernel_name}/`: Individual kernel results
- `attempts_{n}/`: Feedback loop attempt number for each kernel
- `{kernel_name}_data.json`: This holds the feedback data and important data collected in the tools


## Creating New Kernels
Understandably, we know that there will be a desire to create new benchmark kernels. 
The kernels each have one file `generate.py` which contains the following items
### Prompt
The prompt will describe the function to be implemented by the LLM and will briefly describe the inputs and outputs

**EX:** 
```python
description = f"""
This HLS kernel implements a function to calculate the absolute value of a vector of 32-bit integer numbers.
Parameters to the function include two int32_t pointers:
    The first pointer points to the start of a {num_values}-element array of int32_t values to be used as input data.
    The second pointer also points to the start of a {num_values}-element array of int32_t values to be modified and viewed as the output to the function.
The kernel processes each element of the input buffer, computes its absolute value, and stores the result in the corresponding position of the output buffer.
"""
```
### Inputs and Outputs
Each kernel has a `inputs_info` python list which holds the meta data for each parameter of the function to be implemented

**EX:**
```python
inputs_info = [
    {
        "name": "in_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int32",
        "iotype": "in",
        "min": -2**31,
        "max": 2**31 - 1,
    },
    {
        "name": "out_buffer",
        "array_dimension": 1,
        "shape": (num_values,),
        "total_elements": num_values,
        "dtype": "int32",
        "iotype": "out",
    }
]
```

### Behavioral Function
Each kernel generate has a behavioral function that will take in the same inputs as the function itself and then return the outputs in order to create the expected outputs for the testbench.

**EX:** 
```python
def behavioral(input_vector):
    return np.abs(input_vector).astype(np.int32)
```