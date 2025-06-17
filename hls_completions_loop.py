# This is just a practice file an attempt 
# to get a recursive feedback for improvement


from openai import OpenAI
import json
import pathlib
import subprocess
import argparse
from datetime import datetime
import hlseval


def feedback_prompt(prev_code, feedback):
    return f"""
        Here is my C HLS code for Vitis:

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
    
def run_kernel_attempt(model, test, attempt, workflow, mode, prev_feedback=None, goals=None):
    kernel_name = test["kernel_name"]
    solutions_path = f"./results/solutions/{model}/{kernel_name}/attempts_{attempt+1}"
    pathlib.Path(solutions_path).mkdir(parents=True, exist_ok=True)


    if attempt == 0:
        print(f"\tGenerating base completion for {kernel_name} (Attempt {attempt+1})")
        context_string = "Reference vectorized code:\n"
        prompt_with_context = test["prompt"] + "\n" + context_string
        url = (model_parameters[model])[f"{workflow}_url"]


        response = hlseval.generate_one_completion(
            prompt_with_context,
            workflow=workflow,
            model=model,
            url=url,
            temperature=0,
            top_p=1.0,
        )
        try:
            with open(f"{solutions_path}/{kernel_name}.json", "x") as file:
                json.dump(response, file, indent=4)
        except FileExistsError:
            print(f"{solutions_path}/{kernel_name}.json already exists. Skipping")
            

        try:
            with open(f"{solutions_path}/{kernel_name}.cpp", "x") as file:
                file.write(hlseval.get_kernel_code(test, solutions_path))
        except FileExistsError:
            print(f"{solutions_path}/{kernel_name}.cpp already exists. Skipping")

    else:
        if mode == "correction":
            print(f"Improving failed attempt for {kernel_name} (Attempt {attempt})")

        elif mode == "optimization":
            print(f"Optimizing {kernel_name}")
        prev_solutions_path = f"./results/solutions/{model}/{kernel_name}/attempts_{attempt}"

        with open(f"{prev_solutions_path}/{kernel_name}.cpp", "r") as f:
            prev_code = f.read()

        new_prompt = feedback_prompt(prev_code=prev_code, feedback=prev_feedback)
        url = (model_parameters[model])[f"{workflow}_url"]
        improved_code = hlseval.generate_one_completion(
            new_prompt,
            workflow=workflow,
            model=model,
            url=url,
            temperature=0,
            top_p=1.0,
        )

        try:
            with open(f"{solutions_path}/{kernel_name}.json", "x") as file:
                json.dump(improved_code, file, indent=4)
        except FileExistsError:
            print(f"{solutions_path}/{kernel_name}.json already exists. Skipping")

        try:
            with open(f"{solutions_path}/{kernel_name}.cpp", "x") as file:
                file.write(hlseval.get_kernel_code(test, solutions_path))
        except FileExistsError:
            print(f"{solutions_path}/{kernel_name}.cpp already exists. Skipping")
        
    # Run CSIM    
    try:
        print("--- Starting Vitis Run ---")
        csim_output = hlseval.run_vitis_tools(solutions_path, kernel_name, mode="csim")
        timeout_csim = False
        sim_returncode = csim_output.returncode
    except hlseval.CsimTimeoutError as e:
        timeout_csim = True
        sim_returncode = 1
        print("WARNING: Vitis timed out on Simulation")

    if csim_output.returncode == 0:
    # Run Synthesis only if CSIM was successful
        try:
            print("--- Running CSYNTH ---")
            synth_output = hlseval.run_vitis_tools(solutions_path, kernel_name, mode="csynth")
            vitis_feedback = hlseval.gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
            synth_returncode = synth_output.returncode
        except hlseval.CsynthTimeoutError as e:
            print("WARNING: Vitis timed out")
            vitis_feedback = hlseval.gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
            vitis_feedback['synth_feedback']['errors'].extend(["Failed to complete Synthesis. Synthesis took too long due to exploding instructions and resources utilization"])
            vitis_feedback['synth_feedback']['goals'].extend([
                "Try pipelining inside of loops rather than outside to ensure loop pipelining not function pipelining.",
                "Do not unroll completely or partition complete arrays. This may include avoiding pipeline directives."])
            synth_returncode = 1
    else:
        vitis_feedback = hlseval.gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
        vitis_feedback['csim_feedback']['errors'].extend(["Failed to complete Csim. Csim did not run successfully."])
        vitis_feedback['synth_feedback']['errors'].extend(["Synthesis not run due to failed C Simulation."])
        synth_returncode = 1   
    
    print("--- Vitis Run Completed ---")
    returncode = (sim_returncode and synth_returncode) if sim_returncode else synth_returncode
    return returncode, vitis_feedback


def process_kernel(model, test, workflow, passes=3, mode="optimization", goals=None):
    kernel_name = test["kernel_name"]
    prev_ret_code = -1
    testbench_status = "failure"
    testbench_match_status = ""
    feedback = None
    timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    print(f"\nStarting Kernel {kernel_name} at {timestamp}")


    for attempt in range(passes):
        print(f"[{model}] Attempt {attempt+1} for kernel {kernel_name} in mode: {mode}")

        dir_path = pathlib.Path(f"./results/solutions/{model}/{kernel_name}/attempts_{attempt+1}")
        if dir_path.exists() and mode == "correction" and attempt == 0:
            print(f"{kernel_name} already tested and exists. Skipping...\n")
            return

        if attempt == 2 and mode == "correction" and prev_ret_code != 0:
            print(f"Too many failed attempts. Skipping {kernel_name}")
            return

        if mode == "correction" and attempt > 0 and prev_ret_code == 0:
            print(f"Previous attempt successful. Skipping retry.\n")
            return

        # ✅ Pass `goals` only if in optimization mode
        current_goals = goals if mode == "optimization" else None
        prev_ret_code, feedback = run_kernel_attempt(model, test, attempt, workflow, mode, feedback, goals=current_goals)
        print(feedback) # TODO: Feedback has EVERY failing test case I think. Need to cut this down or it will fill the context window.

        if prev_ret_code == 0 and testbench_status == "success" and testbench_match_status == "match":
            print(f"{kernel_name} compiled successfully on attempt {attempt+1}.")
            if mode == "correction":
                return
        timestamp = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        print(f"[{model}] Attempt {attempt+1} for kernel {kernel_name} ended at {timestamp}")


    print(f"All attempts completed for {kernel_name}.")


if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Run HLS kernel processing with optional goals.")

    parser.add_argument(
        'workflow', 
        choices=['amd', 'byu'],  
        help="Which workflow to follow for API calls ('amd' or 'byu')"
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
    
    # Add any models you want completions run for to this list
    # models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    models = ["gpt-4o"]
    passes = 6
    mode = args.mode # Passed in on command line with --mode flag
    goals = args.goals  # Will be None if not provided
    
    with open("./dataset/hlseval.jsonl", "r") as f:
        tests = [json.loads(line) for line in f]

    for model in models:
        if model not in model_parameters:
            print(f"Model {model} not recognized.")
            continue

        for test in tests:
            process_kernel(model, test, workflow=args.workflow, passes=passes, mode=mode, goals=goals)



