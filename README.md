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
Exit the docker container

### Run Vitis Completions
Sample of run command:\
`make run_vitis_completions LOG=DEBUG ATTEMPTS=5`\
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `ATTEMPTS` option: Allows you to determine the number of generation attempts per kernel. Setting this to `1` would make an LLM call for each kernel without incorporating feedback. Setting `ATTEMPTS` to a value of `2` or more will incorporate a feedback loop where the results from each attempt are passed back to the LLM to improve upon the result each time.

### Run DynaRapid Completions
Sample of run command:\
`make run_dynarapid_completions LOG=DEBUG ATTEMPTS=4 RAPIDWRIGHT=TRUE`\
Options:
- `LOG` option: Allows you to set the logging level of the ouput log file. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`
- `ATTEMPTS` option: Allows you to determine the number of generation attempts per kernel. Setting this to `1` would make an LLM call for each kernel without incorporating feedback. Setting `ATTEMPTS` to a value of `2` or more will incorporate a feedback loop where the results from each attempt are passed back to the LLM to improve upon the result each time.
- `RAPIDWRIGHT` option: If omitted, or set to `FALSE`, Dynamatic will run without RapidWright, resulting in a `.dot` file, but no placed and routed circut. Adding the `DYNAMATIC` option and setting it to anything other than `FALSE` will run both Dynamatic and RapidWright. The value `TRUE` is used here for clarity.