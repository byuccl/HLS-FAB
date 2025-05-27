import openai
import subprocess
import tempfile
import re
import requests
import os

# Python Prompt
# SYSTEM_PROMPT = """You are a part of a code generation system for python.
# Your job is to write Python code for a single kernel that will be run on a standard CPU..
# Produce only the python code for the requested kernel including any required headers and imports.
# Make sure the Python code is complete and self contained in a single code block.
# Produce only the kernel function, no main, no additional examples or code.
# """

# HLS C++ prompt
SYSTEM_PROMPT = """You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Vitis HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Make this code so that it is Deterministic when synthesized.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Most kernels are dependent on the standard integer c library. As such, that header almost always needs to be added.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly.
Make sure the C++ code is complete and self contained in a single code block.
Produce only the kernel function, no main, no additional examples or code.
"""

api_key = os.environ["API_KEY"] # reads for the environment variable 'API_KEY'

# The openAI has a different header format than the AMD API.
byu_headers = { 
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
}

amd_headers = { 
    "Ocp-Apim-Subscription-Key": api_key
} 


class HLSCoder:
    """A basic agent that talks via OpenAI APIs"""
    def __init__(
        self, url, workflow, model="gpt-4o", temperature=0.0, top_p=1.0, attempts=1, system_prompt=SYSTEM_PROMPT
    ):
        self.system_prompt=system_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.model = model
        self.temperature = temperature
        self.top_p = top_p
        self.attempts = attempts
        self.workflow = workflow
        self.token_usage = {
            "completion_tokens": 0,
            "prompt_tokens": 0,
            "total_tokens": 0,
        }
        # self.client = openai.OpenAI(base_url=f"{server}/azure/engines/{deployment_id}/chat/completions")

        self.url=url
        if self.workflow == "byu":
            self.headers = HEADERS = byu_headers 
        elif self.workflow == "amd":
            self.headers = HEADERS = amd_headers   



    def __call__(self, prompt):
        """An HLSCoder instance can be called with a user prompt. If self.attempts = 1 it will do a one-shot
        response based on the prompt, if self.attempts > 1 it will check if the generated output compiles and
        retry self.attempts number of times.
        """
        self.messages.append({"role": "user", "content": prompt})

        if self.attempts == 1:
            response = self.generate_code()
            return {
                "response": response,
                "attempt": 0,
                "token_usage": self.token_usage,
                "history": self.messages,
            }

        for attempt in range(self.attempts):
            response = self.generate_code()
            code = self.extract_codeblock(response)

            if not code:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Expected a single codeblock but no code provided.",
                    }
                )
                continue

            compilation_result = compiler_check(code)
            # compiler_check returns a string, we will only see a 'Compilation successful' if the subprocess returned 0
            compile_pass = (
                compilation_result.split("\n")[0] == "Compilation successful."
            )
            if compile_pass:
                return {
                    "response": response,
                    "attempt": attempt,
                    "token_usage": self.token_usage,
                    "history": self.messages,
                }
            else:
                self.messages.append(
                    {
                        "role": "user",
                        "content": f"Compilation failed with:\n{compilation_result}",
                    }
                )

        return {
            "response": response,
            "attempt": self.attempts,
            "token_usage": self.token_usage,
            "history": self.messages,
        }

    def generate_code(self):
        """Lightweight wrapper to generate a response using OpenAI API.
        To learn more about chat completion parameters refer to official docs: https://platform.openai.com/docs/guides/text-generation
        """
        # print(f"\n\n{self.messages}\n\n")
        if self.workflow == "byu":
            body = {
                "model": self.model, # The model is only needed for the byu API
                "messages": self.messages,
                "temperature": self.temperature,
                "top_p": self.top_p
            }
        elif self.workflow == "amd":
            body = {
                "messages": self.messages,
                "temperature": self.temperature,
                "top_p": self.top_p
            }

        # Experiment for the format of claude3
        # body = {
        #     "messages": [{
        #             "role": "user",
        #             "content": [{
        #                     "type": "text",
        #                     "text": "write me a haiku about Australia"
        #                 }
        #             ]
        #         }
        #     ],
        #     "max_tokens": 256,
        #     "temperature": 1
        # }
        response = requests.post(url=self.url,
                         json=body,
                         headers=self.headers)
        # self.update_tokens(response)
        
        # Extract the content of the result
        chat_completion_result = response.json()
        
        if response.status_code != 200:
            print(f"Error: {response.status_code} - {response.text}")
            # Handle the error as needed
            raise requests.exceptions.HTTPError(f"Status {response.status_code}: {response.text}")        
        # response.raise_for_status()
        # chat_completion_result = response.json()

        chat_completion_result = chat_completion_result["choices"]
        chat_completion_result = chat_completion_result[0]
        chat_completion_result = chat_completion_result["message"]
        chat_completion_result = chat_completion_result["content"]

        # print (chat_completion_result)
        self.messages.append({"role": "assistant", "content": chat_completion_result})
        return chat_completion_result

    def update_tokens(self, response):
        """Utility function to get cumulative token counts for cases where the
        agent is making multiple compiler calls and regenerating solutions in a loop.
        """
        usage = response.usage.model_dump()
        self.token_usage["completion_tokens"] += usage["completion_tokens"]
        self.token_usage["prompt_tokens"] += usage["prompt_tokens"]
        self.token_usage["total_tokens"] += usage["total_tokens"]

    @staticmethod
    def extract_codeblock(text):
        """LLMs frequently return code in markdown codeblocks, this function uses regex
        to extract the actual generated code snippet.
        """
        code_blocks = re.findall(
            r"```(?:[a-zA-Z0-9]+)?\n(.*?)```|```(.*?)```", text, re.DOTALL
        )
        # Flatten the list and filter out empty matches
        code_blocks = [block for match in code_blocks for block in match if block]
        return code_blocks[0].strip() if code_blocks else None



def compiler_check(code):
    # Save code to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".cpp", delete=False) as temp_file:
        temp_file.write(code.encode())
        temp_path = temp_file.name

    try:
        # Compile using Vitis HLS (v++) or g++
        result = subprocess.run(
            ["g++", "-c", temp_path, "-o", temp_path + ".o"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return "Compilation successful.\n" + result.stdout
        else:
            return "Compilation failed.\n" + result.stderr
    finally:
        os.remove(temp_path)
        if os.path.exists(temp_path + ".o"):
            os.remove(temp_path + ".o")