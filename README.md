# HLSEval
## Getting Started
### Prerequisites
- Have docker installed (used when running DynaRapid)
- Have python 3.10 or newer
- Have Vitis installed and sourced (We use version 2024.2)
- Set api_key environment variable:
`export OPENROUTER_API_KEY=<Add API Key>`

### Clone and prepare the repo 
```
git clone git@github.com:Preston-Walker/hlseval.git
cd hlseval
python3.10 -m venv venv
source venv/bin/activate
pip install -e .
bash build_rag_dir.sh
```

### Prep for run
```
cd dataset
# Add just one kernel for the test run
cp -r examples/abs_int8 kernels
make dataset
cd ..
```
To set which LLM models to use, edit the `dataset/llm_models.txt` file.\

#### Set up docker container for dynarapid
```
wget https://gist.githubusercontent.com/clavin-xlnx/7ff0b7238aa5f5de4ff2dac7392d5132/raw/a130d502374cfa2779e59f918a2248e61fec5363/Dockerfile.dynarapid
docker build -t my-llvm6-dynarapid-image .
docker run -v <Path to hlseval reposotory>/hlseval:/mount --name dynarapid_container -t my-llvm6-dynarapid-image
```

### Generate Vision Kernel Dataset (Claude Code Skill)
The `/generate_vision_kernel` skill uses Claude Code to generate `generate.py` files for kernels from the [Xilinx Vitis Vision Library](https://github.com/xilinx/vitis_libraries). Each `generate.py` describes a kernel's function signature, I/O buffers, a pure-numpy behavioral model, and example data for AI-assisted HLS code generation.

#### Additional prerequisites
- Clone the Vitis Libraries repository:
```
git clone https://github.com/xilinx/vitis_libraries.git
```
- Have [Claude Code](https://docs.anthropic.com/en/docs/claude-code) installed

#### Configure paths
The skill needs to know where Vitis Libraries and Vitis HLS are on your system. You can configure this in one of two ways:

**Option A — Environment variables** (take priority):
```
export VITIS_LIBRARIES=/path/to/Vitis_Libraries
export VITIS_HLS=/path/to/Vitis_HLS/2024.2
```

**Option B — Config file**:
```
cp vision_skill_paths.cfg.example vision_skill_paths.cfg
# Edit vision_skill_paths.cfg with your paths
```

#### Usage
From within Claude Code, run the skill with a kernel name, `all`, or a path to a list file:
```
/generate_vision_kernel boxfilter
/generate_vision_kernel all
/generate_vision_kernel @kernels_todo.txt
```
Generated files are saved to `dataset/kernels/<top_function_name>/generate.py`.

### Run Vitis Completions
Sample of run command:\
`make run_vitis_completions LOG=DEBUG ATTEMPTS=5`\
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `ATTEMPTS` option: Allows you to determine the number of generation attempts per kernel. Setting this to `1` would make an LLM call for each kernel without incorporating feedback. Setting `ATTEMPTS` to a value of `2` or more will incorporate a feedback loop where the results from each attempt are passed back to the LLM to improve upon the result each time.

### Run DynaRapid Completions
Sample of run command:\
`make run_dynarapid_completions LOG=DEBUG ATTEMPTS=4 RAPIDWRIGHT=TRUE`
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `ATTEMPTS` option: Allows you to determine the number of generation attempts per kernel. Setting this to `1` would make an LLM call for each kernel without incorporating feedback. Setting `ATTEMPTS` to a value of `2` or more will incorporate a feedback loop where the results from each attempt are passed back to the LLM to improve upon the result each time.
- `RAPIDWRIGHT` option: If omitted, or set to `FALSE`, Dynamatic will run without RapidWright, resulting in a `.dot` file, but no placed and routed circut. Adding the `DYNAMATIC` option and setting it to anything other than `FALSE` will run both Dynamatic and RapidWright. The value `TRUE` is used here for clarity.