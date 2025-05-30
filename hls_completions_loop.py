# This is just a practice file an attempt 
# to get a recursive feedback for improvement


from openai import OpenAI
import json
import pathlib
import subprocess
import argparse
from datetime import datetime
from hlseval import HLSCoder, get_kernel_code, gather_parsed_data, test_kernel, NumpyEncoder



class VitisTimeoutError(Exception):
    """Raised when Vitis HLS times out."""
    pass


def generate_one_completion(
    prompt, url, workflow, model="gpt-4o", temperature=0.0, top_p=1.0, attempts=1
):
    
    coder = HLSCoder(
        model=model,
        workflow=workflow,
        temperature=temperature,
        top_p=top_p,
        url=url,
        attempts=attempts,
    )
    response = coder(prompt)

    result = {
        "code": coder.extract_codeblock(response["response"]),
        "stats": {
            "token_usage": response["token_usage"],
            "history": response["history"],
        },
    }

    return result

# Run vitis tools on the generated code
# This function will run the vitis tools on the generated code
def run_vitis_tools(filepath, kernel_name):
    with open("vitis_config.cfg", "w") as file:
        file.write(
            f"""
        part=xazu1eg-sbva484-1-i

        [hls]
        flow_target=vivado
        package.output.format=ip_catalog
        package.output.syn=false
        syn.top={kernel_name}
        syn.file={filepath}{kernel_name}.cpp
        """
        )
    try:
        result = subprocess.run(
            [
                "v++",
                "-c",
                "--mode",
                "hls",
                "--config",
                "vitis_config.cfg",
                "--work_dir",
                filepath,
            ], 
            timeout=200,
            capture_output=True, 
            text=True
        )
        return result
    except subprocess.TimeoutExpired:
        raise VitisTimeoutError("Vitis tools stalled during HLS Synthesis.")
    

def feedback_prompt(prev_code, feedback):
    return f"""
        Here is my C HLS code for Vitis:

        {prev_code}

        Here is some feedback from the synthesis report and the testbench:

        {json.dumps(feedback, indent=2)}

        Please modify the code using the goals from the feedback as a guideline.
        Prioritize fixing errors from either the testbench or the synthesis above all other goals.
        """

byu_gpt_url = "https://api.openai.com/v1/chat/completions"
amd_gpt_url = "https://llm-api.amd.com/openai/deployments/swe-gpt4o-exp1/chat/completions"

model_parameters = {
                        "gpt-4o": {"amd_url": amd_gpt_url, "byu_url": byu_gpt_url},
                        "Claude-3.5": {"url": "https://llm-api.amd.com/claude3/Claude-3.5/chat/completions"}
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


        response = generate_one_completion(
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
                file.write(get_kernel_code(test, solutions_path))
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
        improved_code = generate_one_completion(
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
                file.write(get_kernel_code(test, solutions_path))
        except FileExistsError:
            print(f"{solutions_path}/{kernel_name}.cpp already exists. Skipping")
    try:
        print("Starting Vitis Run")
        vitis_output = run_vitis_tools(f"{solutions_path}/", kernel_name)
        vitis_feedback = gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
    except VitisTimeoutError:#FIXME: What to do when there is a timeout error?
        print("WARNING: Vitis timed out")
        vitis_feedback = gather_parsed_data(kernel_name, model, attempt=attempt+1, goals=goals)
        vitis_feedback['errors'].extend(["Failed to complete Synthesis. Synthesis took too long due to exploding instructions and resources utilization"])
        vitis_feedback['goals'].extend(["Do not unroll completely or partition complete arrays."])
        return 1, vitis_feedback

    result_path = f"{model}/{kernel_name}/attempts_{attempt+1}"
    testbench_output = test_kernel(kernel_name=kernel_name, model_dir=result_path, seed=3)
    testbench_output_path = f"results/solutions/{result_path}/test_results.json"
    with open(testbench_output_path, 'w', encoding='utf-8') as f:  
            json.dump(testbench_output, f, indent=2, cls=NumpyEncoder)  
    print(f"\nResults saved to {testbench_output_path}")  
    testbench_status = testbench_output[kernel_name]["status"]
    testbench_match_status = testbench_output[kernel_name]["match_status"]
    testbench_info = testbench_output[kernel_name]["testbench_info"]
    testbench_feedback = f"""
    Testbench feedback:\n
    C++ Compilation: {testbench_status}
    Match status: {testbench_match_status}
    Testbench info: {testbench_info}
    """
    feedback = f"""
    Vitis feedback:\n
    {vitis_feedback}\n\n
    {testbench_feedback}
    """
    return vitis_output.returncode, feedback


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
        print(feedback)

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
    passes = 3
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



