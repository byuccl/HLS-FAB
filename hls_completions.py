import json
import pathlib
import subprocess
import argparse
from hlseval import HLSCoder, get_kernel_code, gather_parsed_data


#Call the HLSCoder class to get the code from the LLM

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
        capture_output=True, 
        text=True
    )

    return result

def feedback_prompt(prev_code, feedback):
    return f"""
        Here is my C HLS code for Vitis:

        {prev_code}

        Here is some feedback from the synthesis report:

        {json.dumps(feedback, indent=2)}

        Please modify the code using the goals from the feedback as a guideline.
        """

byu_gpt_url = "https://api.openai.com/v1/chat/completions"
amd_gpt_url = "https://llm-api.amd.com/openai/deployments/swe-gpt4o-exp1/chat/completions"

model_parameters = {
                        "gpt-4o": {"amd_url": amd_gpt_url, "byu_url": byu_gpt_url},
                        "Claude-3.5": {"amd_url": "https://llm-api.amd.com/claude3/Claude-3.5/chat/completions"}
                    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="A program to generate HLS kernels using LLMs")  
    parser.add_argument('workflow', choices=['amd', 'byu'],  
                        help="Which workflow to follow for API calls ('amd' or 'byu')")  
    args = parser.parse_args()  
    workflow = args.workflow
    # Add any models you want completions run for to this list
    # models = ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    models = ["gpt-4o"]

    # Needed for LLM completion
    # deployment_id="swe-gpt4o-exp1" # Need to make some kind of lookup table to connect models to deployment ids
    # server="https://llm-api.amd.com/azure"

    with open("./dataset/hlseval.jsonl", "r") as f:
        tests = [json.loads(line) for line in f]

    passes = [1]
    # Loop through the models and kernels for number of attempts to get multiple completions
    for model in models:
        if model in model_parameters:
            if workflow == "byu":
                url = (model_parameters[model])["byu_url"]
            elif workflow == "amd":
                url = (model_parameters[model])["amd_url"]
            else: 
                print("Invalid workflow value. Please select either 'amd', or 'byu'")
            pass
        else:
            print("Model not in the list of available options")
            continue
        for num_attempts in passes:
            for test in tests:
                solutions_path = f"./results/solutions/{model}_attempts_{num_attempts}/{test['kernel_name']}"
                pathlib.Path(solutions_path).mkdir(parents=True, exist_ok=True)
                context_string = "Reference vectorized code:\n"
                prompt_with_context = test["prompt"] + "\n" + context_string

                # Get a single completion
                response = generate_one_completion(
                    prompt_with_context,
                    workflow=workflow,
                    model=model,
                    attempts=num_attempts,
                    temperature=0,
                    top_p=1.0,
                    url=url
                )
                # Print the path the file is being saved to
                print(f"{solutions_path}/{test['kernel_name']}.json")

                # Write all json data to a file
                try:
                    with open(
                        f"{solutions_path}/{test['kernel_name']}.json", "x"
                    ) as file:
                        json.dump(response, file, indent=4)
                except FileExistsError:
                    print(
                        f"The file {solutions_path}/{test['kernel_name']}.json already exists. Skipping file."
                    )

                # Write just the code to a file
                try:
                    with open(
                        f"{solutions_path}/{test['kernel_name']}.cpp", "x"
                    ) as file:
                        file.write(get_kernel_code(test, solutions_path))
                except FileExistsError:
                    print(
                        f"The file {solutions_path}/{test['kernel_name']}.cpp already exists. Skipping file."
                    )

                # Run v++ on the output of the LLM
                result = run_vitis_tools(f"{solutions_path}/", test["kernel_name"])
                if result.returncode != 0:
                    print(f"\n\nDid not Compile: Attempt {num_attempts}\n\n")
                else:
                    print(f"\n\nCompiled Successfully: Attempt {num_attempts}\n\n")
