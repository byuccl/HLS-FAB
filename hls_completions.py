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
from hlseval import HLSEval
import hlseval.utils as utils
import hlseval.dynarapid as dynarapid
from hlseval.prompts import VITIS_LIBRARY_SYSTEM_PROMPT, VITIS_SYSTEM_PROMPT, DYNARAPID_SYSTEM_PROMPT, SYSTEM_PROMPTS, feedback_prompt_dyarapid, feedback_prompt_vitis

logger = logging.getLogger("COMPLETIONS")
timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
# logging.basicConfig(filename=f"results/logs/completions_{timestamp}.log", encoding='utf-8', level=logging.DEBUG)


# A single attempt to generate, test, and synthesize a kernel
def run_kernel_attempt(
    kernel_object: HLSEval,
    attempt: int, 
    prev_feedback: dict=None, 
    ) -> tuple :

    llm_elapsed_time = None
    dynamatic_elapsed_time = None
    csim_elapsed_time = None
    csynth_elapsed_time = None

    llm_elapsed_time = kernel_object.llm_call(attempt, feedback=prev_feedback)


    if kernel_object.hls_flow == "vitis":
        returncode, feedback, csim_elapsed_time, csynth_elapsed_time = kernel_object.vitis_run_w_feedback(attempt)

    elif kernel_object.hls_flow == "dynarapid":
        returncode, feedback, csim_elapsed_time, dynamatic_elapsed_time = dynarapid.run_dynarapid_w_feedback(kernel_object, attempt)

    utils.add_table_row(
        kernel_object._logfile,
        kernel_object.kernel_name, 
        attempt,
        llm_elapsed_time, 
        dynamatic_elapsed_time, 
        csim_elapsed_time, 
        csynth_elapsed_time
    )
    return returncode, feedback

# A wrapper around the feedback loop process for each set of feedback-dependent generation attempts
def process_kernel(kernel_object: HLSEval):
    prev_ret_code = -1
    testbench_status = "failure"
    testbench_match_status = ""
    feedback = None
    timestamp = datetime.now().strftime("%H:%M:%S")
    logger.debug(f"Kernel Start time: {timestamp}")

    # Use a feedback loop to iterate on a design n times
    for attempt in range(kernel_object._num_attempts):
        logger.info(f"Kernel: {kernel_object.kernel_name} Attempt: {attempt+1}")

        prev_ret_code, feedback = run_kernel_attempt(kernel_object, attempt, feedback)
        logger.debug(feedback)

        if prev_ret_code == 0 and testbench_status == "success" and testbench_match_status == "match":
            logger.debug(f"{kernel_object.kernel_name} compiled successfully on attempt {attempt+1}.")

        timestamp = datetime.now().strftime("%H:%M:%S")
        logger.debug(f"[{kernel_object._model}] Attempt {attempt+1} for kernel {kernel_object.kernel_name} ended at {timestamp}")

    logger.info(f"All attempts completed for {kernel_object.kernel_name}")


if __name__ == "__main__":
    """ 
        The main function for HLSEval. 

        Takes a list of models and kernels as well a specifier for whether to use Vitis HLS or Dynarapid.

        Additionally, accepts parameters for the number of attempts (The number of times to regenerate a kernel with feedback) 
        and k (The number of intependent feedback loops to run).
    """ 
    # Arguments set up
    parser = argparse.ArgumentParser(description="Run HLS kernel processing.")
    parser.add_argument("--log", dest='loglevel', help="Desired loglevel", default="WARNING")
    parser.add_argument("--attempts", dest='attempts', help="Number of results to generate within each feedback loop", default=1)
    parser.add_argument("--hls_flow", type=str, choices=['vitis', 'dynarapid'], help="Whether to process HLS for vitis or dynarapid") 
    parser.add_argument("--rapidwright", action='store_true', help="Add this flag to enable RapidWright") 
    parser.add_argument("--skip_behavioral", action='store_true', help="Add this flag to skip Csim" ) 
    parser.add_argument("-k", type=int, default=1, dest="k", help="pass@k with k number of attempts")
    args = parser.parse_args()  

    # Set up Logging Level
    numeric_level = getattr(logging, args.loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % args.loglevel)
    logging.basicConfig(level=numeric_level)
    date = datetime.now().strftime("%B %d %Y")
    logger.info("--------------------------------------------------------------")
    logger.info(f"-  Beginning HLSEval:    {date}                ")
    logger.info("--------------------------------------------------------------")
    logger.debug(f"HLS Tool Chain: {args.hls_flow}")
    logger.debug(f"Attempts: {args.attempts}")
    logger.debug(f"Pass@k: k = {args.k}")
    logger.debug(f"Rapidwright Enabled: {args.rapidwright}")
    logger.debug(f"Skipping Behavioral: {args.skip_behavioral}")

    passes = int(args.attempts)
    rapidwright = args.rapidwright
    skip_behavioral = args.skip_behavioral


    with open("./dataset/hlseval.jsonl", "r") as f:
        kernels = [json.loads(line) for line in f]

    # Add any models you want completions run for to the file dataset/llm_models.txt
    with open("./dataset/llm_models.txt", "r") as f:
        lines = f.readlines()
    models = [line.strip() for line in lines]

    # Process each individual kernel through each model etc.
    for model in models:
        for iter in range(args.k):
            logger.info("--------------------------------------------------------------")
            pathlib.Path(f"results/solutions/{args.hls_flow}/{model}/{iter+1}/logs/").mkdir(parents=True, exist_ok=True)
            with open (f"results/solutions/{args.hls_flow}/{model}/{iter+1}/logs/timing.log", "w") as logfile:
                hlseval.write_table_header(logfile)
                for kernel in kernels:
                    logger.info(f"Running Model: {model} Kernel: {kernel["kernel_name"]} Pass: {iter+1}")
                    kernel_object = hlseval.HLSEval(model, kernel, args.hls_flow, iter, passes, skip_behavioral, rapidwright, logfile)
                    process_kernel(kernel_object)
                    
                hlseval.write_table_footer(logfile)
 
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






