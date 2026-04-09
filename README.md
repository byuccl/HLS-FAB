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
To set which LLM models to use, edit the `dataset/llm_models.txt` file. We use Openrouter convention for LLM Names. `openai/gpt-4.1-mini`

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
`make run_vitis_completions LOG=DEBUG N=2 K=2`\
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `N` option: Allows you to determine the number of generation attempts per kernel. Setting this to `1` would make an LLM call for each kernel without incorporating feedback. Setting `ATTEMPTS` to a value of `2` or more will incorporate a feedback loop where the results from each attempt are passed back to the LLM to improve upon the result each time.
- `K` option: Allows you to run the process with all feedback attempts `K` times independently.
  - Example: K=3, N=4: Each kernel will be generated a total of 12 times. There will be three independent sets of feedback loops, each loop has four generatation attempts building on feedback from the last attempt.

### Run DynaRapid Completions
Sample of run command:\
`make run_dynarapid_completions LOG=DEBUG N=2 K=2 RAPIDWRIGHT=TRUE`\
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `N` option: Same as above.
- `K` option: Same as above.
- `RAPIDWRIGHT` option: If omitted, or set to `FALSE`, Dynamatic will run without RapidWright, resulting in a `.dot` file, but no placed and routed circut. Adding the `DYNAMATIC` option and setting it to anything other than `FALSE` will run both Dynamatic and RapidWright. The value `TRUE` is used here for clarity.
