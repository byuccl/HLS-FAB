import logging
from openai import OpenAI
import os
import json
import pathlib
import subprocess
import argparse
from datetime import datetime
import hlseval
import csv
from hlseval import run_dynarapid

logger = logging.getLogger("COMPLETIONS")
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logging.basicConfig(filename=f"results/logs/completions_{timestamp}.log", encoding='utf-8', level=logging.DEBUG)

VITIS_SYSTEM_PROMPT = """You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Vitis HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Do not use size as a function argument, instead use a constant defined at the top of the file or function.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Most kernels are dependent on the arbitrary precision integer library for HLS (<ap_int.h>) or the arbitrary precision floating point library (<ap_float.h>). As such, that header almost always needs to be added.
Some data types are represented as these arbitrary precision integers or floating point numbers, for example, a bfloat16 is represented with the type `ap_float<16,8>`.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Produce only the kernel function, no main, no additional examples or code.

IMPORTANT: You MUST use the search_wrapper tool to retrieve relevant code examples before generating any code. Never write code without first consulting the examples in the database.

"""

DYNARAPID_SYSTEM_PROMPT = """
You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Dynamatic HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Make this code so that it is Deterministic when synthesized.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Dynamatic does not support arbitrary precision integers (from the <ap_int.h> library) 
The integer types you are allowed to use are int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.

"""

def write_table_header(logfile):
    """Write the header of the ASCII table to the logfile."""
    # Define headers and column widths
    headers = ["Kernel", "LLM Time", "Dynamatic time", "Vitis CSIM time", "Vitis CSYNTH time"]
    col_widths = [30, 20, 20, 20, 20]
    
    # Create top border
    border = "+"
    for width in col_widths:
        border += "-" * width + "+"
    logfile.write(border + "\n")
    
    # Write header row
    header_row = "|"
    for header, width in zip(headers, col_widths):
        header_row += header.center(width) + "|"
    logfile.write(header_row + "\n")
    
    # Write separator after headers
    logfile.write(border + "\n")
    logfile.flush()

def add_table_row(logfile, kernel, llm_time = "-", dynamatic_time = "-", vitis_csim_time = "-", vitis_csynth_time = "-"):
    """Add a row to the ASCII table in the logfile."""
    col_widths = [30, 20, 20, 20, 20]
    
    # Prepare the data
    data = [kernel, llm_time, dynamatic_time, vitis_csim_time, vitis_csynth_time]
    
    # Create the data row
    data_row = "|"
    for value, width in zip(data, col_widths):
        # Convert to string and center in the column
        value_str = str(value) if value is not None else ""
        data_row += value_str.center(width) + "|"
    
    logfile.write(data_row + "\n")
    logfile.flush()

def write_table_footer(logfile):
    """Write the footer (bottom border) of the ASCII table to the logfile."""
    col_widths = [30, 20, 20, 20, 20]
    
    # Create bottom border
    border = "+"
    for width in col_widths:
        border += "-" * width + "+"
    
    logfile.write(border + "\n")
    logfile.flush()

def feedback_prompt(prev_code, feedback):
    return f"""
Here is my C HLS code:

{prev_code}

Here is some feedback from the synthesis report and the testbench:

{json.dumps(feedback, indent=2)}

Please modify the code using the goals from the feedback as a guideline.

Prioritize the goals to improve the code in the following order:
TIER 1 - CRITICAL ERRORS (Must Fix First):
- Compilation failures (status: failed)
- Synthesis failures (latency: NULL, resources: NULL, errors: non-empty)  
- Simulation mismatches (percent_error > 0%, match: mismatch)
- Function returns non-zero

TIER 2 - PERFORMANCE WARNINGS (Address if Tier 1 is clear):
- II Violations → Memory port contention
- Resource overflow → Exceeding FPGA limits (BRAM: 216, DSP: 216, FF: 74880 , LUT: 37440)
- Timing violations → Clock frequency issues

TIER 3 - OPTIMIZATION OPPORTUNITIES (Final stage):
- Latency reduction
- Resource efficiency
- Throughput improvement

Analyze the C Simulation and C Synthesis feedback to determine whether the code meets the first and second goals before moving on to the third goal of optimization
"""

byu_gpt_url2 = "https://api.openai.com/v1/completions"


byu_gpt_url = "https://api.openai.com/v1/chat/completions"
amd_gpt_4o_url = "https://llm-api.amd.com/openai/deployments/swe-gpt4o-exp1/chat/completions"
amd_gpt_4_1_url = "https://llm-api.amd.com/azure/engines/dvue-aoai-001-gpt-4.1/chat/completions"
amd_llama_3_2_url = "https://llm-api.amd.com/api/OnPrem/openai/deployments/Llama-3.2-11B-Vision/chat/completions?api-version=2023-05-15"
byu_llama_url = "https://api.llama.com/v1/chat/completions"
amd_deepseek_r1_url = "https://llm-api.amd.com/api/OnPrem/openai/deployments/deepseek-r1/chat/completions?api-version=2023-05-15"
byu_antrhopic_url = "https://api.anthropic.com/v1/complete"
byu_anthropic_url2 = "https://api.anthropic.com/v1/messages"
amd_claude_3_5_url = "https://llm-api.amd.com/claude3/Claude-3.5/chat/completions"
amd_gemini_2_url = "https://llm-api.amd.com/vertex/gemini/deployments/gemini-2.0-flash-001/chat/completions"
openrouter_url = "https://openrouter.ai/api/v1/chat/completions"


model_parameters = {
                        "gpt-4o": {"amd_url": amd_gpt_4o_url, "byu_url": byu_gpt_url},
                        "gpt-4.1": {"amd_url": amd_gpt_4_1_url, "byu_url": byu_gpt_url},
                        "Claude-3.5": {"amd_url": amd_claude_3_5_url, "byu_url": byu_anthropic_url2},
                        "Llama-3_2-11B": {"amd_url": amd_llama_3_2_url, "byu_url": byu_gpt_url},
                        "Deepseek-R1": {"amd_url": amd_deepseek_r1_url, "byu_url": byu_gpt_url},
                        "Gemini-2.0-Flash": {"amd_url": amd_gemini_2_url, "byu_url": byu_gpt_url}
                    }
    
def run_kernel_attempt(model, test, attempt, workflow, hls_flow, rapidwrite, mode, prev_feedback=None, goals=None):
    kernel_name = test["kernel_name"]
    llm_elapsed_time = None
    dynamatic_elapsed_time = None
    csim_elapsed_time = None
    csynth_elapsed_time = None
    solutions_path = f"./results/solutions/{hls_flow}/{model}/{kernel_name}/attempts_{attempt+1}"
    pathlib.Path(solutions_path).mkdir(parents=True, exist_ok=True)


    if attempt == 0:
        logger.debug(f"\tGenerating base completion for {kernel_name} (Attempt {attempt+1})")

        context_string = "Reference vectorized code:\n"
        prompt_with_context = test["prompt"] + "\n" + context_string
        if workflow == "openrouter":
            # logger.debug("Setting openrouter url")
            url = openrouter_url
        else:
            url = (model_parameters[model])[f"{workflow}_url"]

        llm_completion_begin = datetime.now()

        response = hlseval.generate_one_completion(
            prompt_with_context,
            workflow=workflow,
            hls_flow=hls_flow,
            model=model,
            url=url,
            system_prompt=VITIS_SYSTEM_PROMPT if hls_flow=="vitis" else DYNARAPID_SYSTEM_PROMPT,
            temperature=0,
            top_p=1.0,
            rag=True if hls_flow=="vitis" else False
        )
        llm_completion_end = datetime.now()
        llm_elapsed_time = llm_completion_end - llm_completion_begin
        logger.debug(f"LLM API call elapsed time: {llm_elapsed_time}")

        try:
            with open(f"{solutions_path}/{kernel_name}.json", "x") as file:
                json.dump(response, file, indent=4)
        except FileExistsError:
            logger.info(f"{solutions_path}/{kernel_name}.json already exists. Skipping")
            

        try:
            with open(f"{solutions_path}/{kernel_name}.cpp", "x") as file:
                logger.debug("Writing to file")
                kernel_code = hlseval.get_kernel_code(hls_flow, test, solutions_path)
                if kernel_code == None:
                        add_table_row(
                            logfile,
                            kernel_name, 
                            llm_elapsed_time, 
                            dynamatic_elapsed_time, 
                            csim_elapsed_time, 
                            csynth_elapsed_time
                        )
                        return 1, "Failed to produce kernel code"
                else:
                    file.write(kernel_code)
        except FileExistsError:
            logger.info(f"{solutions_path}/{kernel_name}.cpp already exists. Skipping")

    else:
        if mode == "correction":
            logger.debug(f"Improving failed attempt for {kernel_name} (Attempt {attempt})")

        elif mode == "optimization":
            logger.debug(f"Optimizing {kernel_name}")
        prev_solutions_path = f"./results/solutions/{hls_flow}/{model}/{kernel_name}/attempts_{attempt}"

        with open(f"{prev_solutions_path}/{kernel_name}.cpp", "r") as f:
            prev_code = f.read()

        llm_completion_begin = datetime.now()

        new_prompt = feedback_prompt(prev_code=prev_code, feedback=prev_feedback)
        logger.debug(new_prompt)
        if workflow == "openrouter":
            # logger.debug("Setting openrouter url")
            url = openrouter_url
        else:
            url = (model_parameters[model])[f"{workflow}_url"]
        improved_code = hlseval.generate_one_completion(
            new_prompt,
            workflow=workflow,
            hls_flow=hls_flow,
            model=model,
            url=url,
            system_prompt=VITIS_SYSTEM_PROMPT if hls_flow=="vitis" else DYNARAPID_SYSTEM_PROMPT,
            temperature=0,
            top_p=1.0,
            rag=True if hls_flow=="vitis" else False
        )
        llm_completion_end = datetime.now()
        llm_elapsed_time = llm_completion_end - llm_completion_begin
        logger.debug(f"LLM API call elapsed time: {llm_elapsed_time}")
        try:
            with open(f"{solutions_path}/{kernel_name}.json", "x") as file:
                json.dump(improved_code, file, indent=4)
        except FileExistsError:
            logger.info(f"{solutions_path}/{kernel_name}.json already exists. Skipping")

        try:
            with open(f"{solutions_path}/{kernel_name}.cpp", "x") as file:
                logger.debug("Writing to file")
                kernel_code = hlseval.get_kernel_code(hls_flow, test, solutions_path)
                if kernel_code == None:
                        add_table_row(
                            logfile,
                            kernel_name, 
                            llm_elapsed_time, 
                            dynamatic_elapsed_time, 
                            csim_elapsed_time, 
                            csynth_elapsed_time
                        )
                        return 1, "Failed to produce kernel code"
                else:
                    file.write(kernel_code)
        except FileExistsError:
            logger.info(f"{solutions_path}/{kernel_name}.cpp already exists. Skipping")
    
    if hls_flow == "vitis":
        # Run CSIM    
        csim_begin = datetime.now()

        try:
            logger.info("Starting Vitis Run")
            csim_output = hlseval.run_vitis_tools(solutions_path, kernel_name, mode="csim")
            timeout_csim = False
            sim_returncode = csim_output.returncode
        except hlseval.CsimTimeoutError as e:
            timeout_csim = True
            sim_returncode = 1
            logger.warning("Vitis timed out on Simulation")
        csim_end = datetime.now()
        csim_elapsed_time = csim_end-csim_begin
        logger.debug(f"CSIM elapsed time: {csim_elapsed_time}")

        if sim_returncode == 0:
        # Run Synthesis only if CSIM was successful
            try:
                logger.info("--- Running CSYNTH ---")
                csynth_begin = datetime.now()

                synth_output = hlseval.run_vitis_tools(solutions_path, kernel_name, mode="csynth")
                vitis_feedback = hlseval.gather_parsed_data(kernel_name, hls_flow, model, attempt=attempt+1, goals=goals)
                synth_returncode = synth_output.returncode
                csynth_end = datetime.now()
                csynth_elapsed_time = csynth_end-csynth_begin
                logger.debug(f"CYNTH elapsed time: {csynth_elapsed_time}")  
            except hlseval.CsynthTimeoutError as e:
                logger.warning(" Vitis timed out")
                vitis_feedback = hlseval.gather_parsed_data(kernel_name, hls_flow, model, attempt=attempt+1, goals=goals)
                logger.debug(vitis_feedback)
                logger.debug("vitis Feedback before Adding new Errors")
                vitis_feedback['synth_feedback']['errors'].extend(["Failed to complete Synthesis. Synthesis took too long due to exploding instructions and resources utilization"])
                vitis_feedback['synth_feedback']['goals'].extend([
                    "Try pipelining inside of loops rather than outside to ensure loop pipelining not function pipelining."
                ])
                logger.debug(vitis_feedback)
                logger.debug("vitis Feedback before Adding new Errors")
                synth_returncode = 1
        else:
            vitis_feedback = hlseval.gather_parsed_data(kernel_name, hls_flow, model, attempt=attempt+1, goals=goals)
            vitis_feedback['sim_feedback']['errors'].extend(["Failed to complete Csim. Csim did not run successfully."])
            vitis_feedback['synth_feedback']['errors'].extend(["Synthesis not run due to failed C Simulation."])
            synth_returncode = 1   

        logger.info("--- Vitis Run Completed ---")

        out_data_path = f"{solutions_path}/{kernel_name}_data.json"
        hlseval.save_feedback_data(out_data_path, vitis_feedback)
        

        returncode = (sim_returncode and synth_returncode) if sim_returncode else synth_returncode
        feedback = vitis_feedback
    elif hls_flow == "dynarapid":
        # Run CSIM    
        dsim_begin = datetime.now()

        try:
            logger.info("Starting Dynarapid Simulation")
            dsim_output = hlseval.run_vitis_tools(solutions_path, kernel_name, mode="csim", hls_flow=hls_flow)
            logger.debug(dsim_output)
            timeout_dsim = False
            sim_returncode = dsim_output.returncode
        except hlseval.CsimTimeoutError as e:
            timeout_dsim = True
            sim_returncode = 1
            logger.warning("Vitis timed out on Simulation")
        dsim_end = datetime.now()
        csim_elapsed_time = dsim_end-dsim_begin
        logger.debug(f"Dynarapid Simulation elapsed time: {csim_elapsed_time}")
        dynarapid_feedback = hlseval.gather_parsed_data(kernel_name, hls_flow=hls_flow, model=model, attempt=attempt+1, goals=goals)
    

        returncode, stdout, dynamatic_elapsed_time, rapidwrite_elapsed_time = run_dynarapid(kernel_name, solutions_path, attempt, model, rapidwrite)

        dynarapid_feedback['compile_feedback'] = {
            "returncode": returncode,
            "stdout": stdout,
        }
        out_data_path = f"{solutions_path}/{kernel_name}_data.json"
        # try:
        #     with open(f"{solutions_path}/{kernel_name}_test_data.json", "r") as f:
        #         print(f"Found test data file for Dynarapid {solutions_path}/{kernel_name}_test_data.json")
        # except FileNotFoundError:
        #     print(f"Did not find test data file for Dynarapid {solutions_path}/{kernel_name}_test_data.json")
        hlseval.save_feedback_data(out_data_path, dynarapid_feedback)
        feedback = dynarapid_feedback



    add_table_row(
        logfile,
        kernel_name, 
        llm_elapsed_time, 
        dynamatic_elapsed_time, 
        csim_elapsed_time, 
        csynth_elapsed_time
    )
    return returncode, feedback


def process_kernel(model, test, workflow, hls_flow, logfile, rapidwrite, passes=3, mode="optimization", goals=None):
    kernel_name = test["kernel_name"]
    prev_ret_code = -1
    testbench_status = "failure"
    testbench_match_status = ""
    feedback = None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    logger.info(f"Starting Kernel {kernel_name} at {timestamp}")


    for attempt in range(passes):
        logger.debug(f"[{model}] Attempt {attempt+1} for kernel {kernel_name} in mode: {mode}")

        dir_path = pathlib.Path(f"./results/solutions/{hls_flow}/{model}/{kernel_name}/attempts_{attempt+1}")
        if dir_path.exists() and mode == "correction" and attempt == 0:
            logger.debug(f"{kernel_name} already tested and exists. Skipping...\n")
            return

        if attempt == 2 and mode == "correction" and prev_ret_code != 0:
            logger.debug(f"Too many failed attempts. Skipping {kernel_name}")
            return

        if mode == "correction" and attempt > 0 and prev_ret_code == 0:
            logger.debug(f"Previous attempt successful. Skipping retry.\n")
            return

        # Pass `goals` only if in optimization mode
        current_goals = goals if mode == "optimization" else None
        prev_ret_code, feedback = run_kernel_attempt(model, test, attempt, workflow, hls_flow, mode, rapidwrite, feedback, goals=current_goals)
        logger.debug(feedback)

        if prev_ret_code == 0 and testbench_status == "success" and testbench_match_status == "match":
            logger.debug(f"{kernel_name} compiled successfully on attempt {attempt+1}.")
            if mode == "correction":
                return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        logger.debug(f"[{model}] Attempt {attempt+1} for kernel {kernel_name} ended at {timestamp}")


    logger.info(f"All attempts completed for {kernel_name}.")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run HLS kernel processing with optional goals.")

    parser.add_argument("--log", dest='loglevel', help="Desired loglevel", default="WARNING")

    parser.add_argument("--attempts", dest='attempts', help="Number of results to generate", default=1)


    parser.add_argument(
        'workflow', 
        choices=['amd', 'byu', 'openrouter'],  
        help="Which workflow to follow for API calls ('amd', 'byu', or 'openrouter')"
    )  

    parser.add_argument(
        "--hls_flow",
        type=str,
        choices=['vitis', 'dynarapid'],  
        help="Whether to process HLS for vitis or dynarapid"
    ) 

    parser.add_argument(
        "--mode",
        type=str,
        choices=["correction", "optimization"],
        default="optimization",
        help="Choose whether to run in correction or optimization mode."
    )

    parser.add_argument(
        "--goals",
        type=str,
        nargs="*",
        help="List of goals for optimization mode (e.g., --goals 'maximize Fmax' 'reduce latency')"
    )
    parser.add_argument(
        "--rapidwrite",
        action='store_true',
        help="Add this flag to enable RapidWrite"
    ) 



    args = parser.parse_args()  

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)
    
    # Add any models you want completions run for to this list
    # models = ["gpt-4o", "gpt-4.1", "Llama-3_2-11B", "Claude-3.5", "Gemini-2.0-Flash", "Deepseek-R1" (Context window so small it is unusable) ]
    models = ["x-ai/grok-4", "openai/gpt-4.1"]
    passes = int(args.attempts)
    mode = args.mode # Passed in on command line with --mode flag
    goals = args.goals  # Will be None if not provided
    if args.rapidwrite:
        rapidwrite = True
    else:
        rapidwrite = False
    logger.debug(f"RapidWrite: {rapidwrite}")

    with open("./dataset/hlseval.jsonl", "r") as f:
        tests = [json.loads(line) for line in f]


    for model in models:
        # if model not in model_parameters:
        #     logger.warning(f"Model {model} not recognized.")
        #     continue

        pathlib.Path(f"results/solutions/{args.hls_flow}/{model}/logs/").mkdir(parents=True, exist_ok=True)
        with open (f"results/solutions/{args.hls_flow}/{model}/logs/timing.log", "w") as logfile:
            write_table_header(logfile)

            for test in tests:
                process_kernel(model, test, workflow=args.workflow, hls_flow=args.hls_flow, logfile=logfile, rapidwrite=rapidwrite, passes=passes, mode=mode, goals=goals)
            
            write_table_footer(logfile)
    # Delete the unecessary files after running the completions
    try:
        os.remove("kernel.json")
        logger.debug(f"File 'kernel.json' deleted successfully.")
    except FileNotFoundError:
        logger.debug(f"Error: File 'kernel.json' not found.")
    except Exception as e:
        logger.debug(f"An error occurred: {e}")
    try:
        os.remove("vitis_config.cfg")
        logger.debug(f"File 'vitis_config.cfg' deleted successfully.")
    except FileNotFoundError:
        logger.debug(f"Error: File 'vitis_config.cfg' not found.")
    except Exception as e:
        logger.debug(f"An error occurred: {e}")






