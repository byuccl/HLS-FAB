# This is just a practice file an attempt 
# to get a recursive feedback for improvement

import logging
from openai import OpenAI
import json
import pathlib
import subprocess
import argparse
from datetime import datetime
import hlseval
from hlseval import run_dynarapid

logger = logging.getLogger("COMPLETIONS")
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
logging.basicConfig(filename=f"results/logs/completions_{timestamp}.log", encoding='utf-8', level=logging.DEBUG)

VITIS_SYSTEM_PROMPT = """You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Vitis HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Do not use size as a function argument, instead use a constant defined at the top of the file or function.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Most kernels are dependent on the arbitrary precision integer library for HLS (<ap_int.h>). As such, that header almost always needs to be added.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Produce only the kernel function, no main, no additional examples or code.

IMPORTANT: You MUST use the search_wrapper tool to retrieve relevant code examples before generating any code. Never write code without first consulting the examples in the database.

In cases where you are asked to improve existing code, Prioritize these goals to improve the code in the following order:
1. Running successfully - The code should successfully compile and run in both C Simulation and C Synthesis (look for phrases such as "Simulation Failed", "Failed to complete Synthesis" or "Function 'main' returns nonzero value")
2. Functional correctness - The code should produce the desired output based on the C Simulation results (look for a series of "Expected" and "Actual" statements indicating that not every output value was correct. Also watch for the "percent_error" value. Anything above 0 is unacceptable.)
3. Optimal performance - Reduce the latency while trying to also keep resource usage low

Analyze the C Simulation and C Synthesis feedback to determine whether the code meets the first and second goals before moving on to the third goal of optimization
"""

DYNARAPID_SYSTEM_PROMPT = """
You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Dynamatic HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Make this code so that it is Deterministic when synthesized.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Dynamatic does not support arbitrary precision integers (from the <ap_int.h> library) or C++ standard integers (from the <stdint.h> library). Therefore, only C++ built-in types may be used.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
IMPORTANT: With each kernel, produce a simple main function to initialize any arrays and call the function. Do not add any kind of print statements to the main function. These will prevent the code from compiling successfully.

In cases where you are asked to improve existing code, Prioritize these goals to improve the code in the following order:
1. Running successfully - The code should successfully compile and run in both C Simulation and C Synthesis (look for phrases such as "Simulation Failed", "Failed to complete Synthesis" or "Function 'main' returns nonzero value")
2. Functional correctness - The code should produce the desired output based on the C Simulation results (look for a series of "Expected" and "Actual" statements indicating that not every output value was correct. Also watch for the "percent_error" value. Anything above 0 is unacceptable.)
3. Optimal performance - Reduce the latency while trying to also keep resource usage low - Note: Unlike many other HLS tools, Dynamatic does not use pragmas to help with optimization. Code restructuring is the only way to better optimize kernels.

Analyze the given feedback to determine whether the code meets the first and second goals before moving on to the third goal of optimization
"""

def feedback_prompt(prev_code, feedback):
    return f"""
        Here is my C HLS code:

        {prev_code}

        Here is some feedback from the synthesis report and the testbench:

        {json.dumps(feedback, indent=2)}

        Please modify the code using the goals from the feedback as a guideline.

        Prioritize the goals to improve the code in the following order:
        1. Running successfully - The code should successfully compile and run in both C Simulation and C Synthesis (look for phrases such as "Simulation Failed" or "Function 'main' returns nonzero value")
        2. Functional correctness - The code should produce the desired output based on the C Simulation results (look for a series of "Expected" and "Actual" statements indicating that not every output value was correct. Also watch for the "percent_error" value. Anything above 0 is unacceptable.)
        3. Optimal performance - Reduce the latency while trying to also keep resource usage low. If the latency says 0 then something is probably wrong with the code. Look at the warnings and errors in the synthesis report to guide your optimizations.
        
        Analyze the C Simulation and C Synthesis feedback to determine whether the code meets the first and second goals before moving on to the third goal of optimization
        """

byu_gpt_url = "https://api.openai.com/v1/chat/completions"
amd_gpt_url = "https://llm-api.amd.com/openai/deployments/swe-gpt4o-exp1/chat/completions"
byu_antrhopic_url = "https://api.anthropic.com/v1/complete"
byu_anthropic_url2 = "https://api.anthropic.com/v1/messages"
amd_anthropic_url = "https://llm-api.amd.com/claude3/Claude-3.5/chat/completions"

model_parameters = {
                        "gpt-4o": {"amd_url": amd_gpt_url, "byu_url": byu_gpt_url},
                        "Claude-3.5": {"amd_url": amd_anthropic_url, "byu_url": byu_anthropic_url2}
                    }
    
def run_kernel_attempt(model, test, attempt, workflow, hls_flow, mode, prev_feedback=None, goals=None):
    kernel_name = test["kernel_name"]
    solutions_path = f"./results/solutions/{hls_flow}/{model}/{kernel_name}/attempts_{attempt+1}"
    pathlib.Path(solutions_path).mkdir(parents=True, exist_ok=True)


    if attempt == 0:
        logger.debug(f"Generating base completion for {kernel_name} (Attempt {attempt+1})")
        context_string = "Reference vectorized code:\n"
        prompt_with_context = test["prompt"] + "\n" + context_string
        url = (model_parameters[model])[f"{workflow}_url"]


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
        try:
            with open(f"{solutions_path}/{kernel_name}.json", "x") as file:
                json.dump(response, file, indent=4)
        except FileExistsError:
            logger.info(f"{solutions_path}/{kernel_name}.json already exists. Skipping")
            

        try:
            with open(f"{solutions_path}/{kernel_name}.cpp", "x") as file:
                logger.debug("Writing to file")
                file.write(hlseval.get_kernel_code(hls_flow, test, solutions_path))
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

        new_prompt = feedback_prompt(prev_code=prev_code, feedback=prev_feedback)
        logger.error(new_prompt)
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

        try:
            with open(f"{solutions_path}/{kernel_name}.json", "x") as file:
                json.dump(improved_code, file, indent=4)
        except FileExistsError:
            logger.info(f"{solutions_path}/{kernel_name}.json already exists. Skipping")

        try:
            with open(f"{solutions_path}/{kernel_name}.cpp", "x") as file:
                logger.debug("Writing to file")
                file.write(hlseval.get_kernel_code(hls_flow, test, solutions_path))
        except FileExistsError:
            logger.info(f"{solutions_path}/{kernel_name}.cpp already exists. Skipping")
    
    if hls_flow == "vitis":
        # Run CSIM    
        try:
            logger.info("Starting Vitis Run")
            csim_output = hlseval.run_vitis_tools(solutions_path, kernel_name, mode="csim")
            timeout_csim = False
            sim_returncode = csim_output.returncode
        except hlseval.CsimTimeoutError as e:
            timeout_csim = True
            sim_returncode = 1
            logger.warning("Vitis timed out on Simulation")

        if csim_output.returncode == 0:
        # Run Synthesis only if CSIM was successful
            try:
                logger.info("Running CSYNTH")
                synth_output = hlseval.run_vitis_tools(solutions_path, kernel_name, mode="csynth")
                vitis_feedback = hlseval.gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
                synth_returncode = synth_output.returncode
            except hlseval.CsynthTimeoutError as e:
                logger.warning("Vitis timed out")
                vitis_feedback = hlseval.gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
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
            vitis_feedback = hlseval.gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
            vitis_feedback['csim_feedback']['errors'].extend(["Failed to complete Csim. Csim did not run successfully."])
            vitis_feedback['synth_feedback']['errors'].extend(["Synthesis not run due to failed C Simulation."])
            synth_returncode = 1   
        
        logger.info("Vitis Run Completed")

        out_data_path = f"{solutions_path}/{kernel_name}_data.json"
        hlseval.save_feedback_data(out_data_path, vitis_feedback)
        

        returncode = (sim_returncode and synth_returncode) if sim_returncode else synth_returncode
        feedback = vitis_feedback
    elif hls_flow == "dynarapid":
        returncode, feedback = run_dynarapid(kernel_name, solutions_path)
    return returncode, feedback


def process_kernel(model, test, workflow, hls_flow, passes=3, mode="optimization", goals=None):
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
        prev_ret_code, feedback = run_kernel_attempt(model, test, attempt, workflow, hls_flow, mode, feedback, goals=current_goals)
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
        choices=['amd', 'byu'],  
        help="Which workflow to follow for API calls ('amd' or 'byu')"
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


    args = parser.parse_args()  

    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)
    
    # Add any models you want completions run for to this list
    # models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    models = ["gpt-4o"]
    passes = int(args.attempts)
    mode = args.mode # Passed in on command line with --mode flag
    goals = args.goals  # Will be None if not provided
    
    with open("./dataset/hlseval.jsonl", "r") as f:
        tests = [json.loads(line) for line in f]

    for model in models:
        if model not in model_parameters:
            logger.warning(f"Model {model} not recognized.")
            continue

        for test in tests:
            process_kernel(model, test, workflow=args.workflow, hls_flow=args.hls_flow, passes=passes, mode=mode, goals=goals)



