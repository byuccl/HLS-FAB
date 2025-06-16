import openai
import subprocess
import tempfile
import re
import requests
import os
import json  

from typing import Dict, Any, List  
  
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings  
from llama_index.core import MockEmbedding  

# Try to import HuggingFaceEmbedding from different possible locations  
try:  
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  
    print("Successfully imported HuggingFaceEmbedding from llama_index.embeddings.huggingface")  
except ImportError:  
    try:  
        from llama_index_embeddings_huggingface import HuggingFaceEmbedding  
        print("Successfully imported HuggingFaceEmbedding from llama_index_embeddings_huggingface")  
    except ImportError:  
        print("Falling back to MockEmbedding")  
        HuggingFaceEmbedding = None  


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

IMPORTANT: You MUST use the search_wrapper tool to retrieve relevant code examples before generating any code. Never write code without first consulting the examples in the database.
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
        self, url, workflow, model="gpt-4o", temperature=0.0, top_p=1.0, attempts=1, system_prompt=SYSTEM_PROMPT, rag=False
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
        self.rag = rag

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
        if self.rag:
            index, embed_model = setup_rag()  

            # Save the index for future use  
            print("Saving the index...")  
            index.storage_context.persist("storage")  
            
            print("Running the RAG agent...")  
            response = rag_agent(  
                self,  
                index  
            )  
        else:
            response = call_llm_api(self)["choices"][0]["message"]["content"] 
        print("\nFinal Response:")  
        print(response)  

        # # print(f"\n\n{self.messages}\n\n")
        # if self.workflow == "byu":
        #     body = {
        #         "model": self.model, # The model is only needed for the byu API
        #         "messages": self.messages,
        #         "temperature": self.temperature,
        #         "top_p": self.top_p
        #     }
        # elif self.workflow == "amd":
        #     body = {
        #         "messages": self.messages,
        #         "temperature": self.temperature,
        #         "top_p": self.top_p
        #     }
        # if tools:  
        #     body["tools"] = tools  

        # # Experiment for the format of claude3
        # # body = {
        # #     "messages": [{
        # #             "role": "user",
        # #             "content": [{
        # #                     "type": "text",
        # #                     "text": "write me a haiku about Australia"
        # #                 }
        # #             ]
        # #         }
        # #     ],
        # #     "max_tokens": 256,
        # #     "temperature": 1
        # # }
        # response = requests.post(url=self.url,
        #                  json=body,
        #                  headers=self.headers)
        # self.update_tokens(response)
        
        # Extract the content of the result
        # chat_completion_result = response.json()
        
        # if response.status_code != 200:
        #     print(f"Error: {response.status_code} - {response.text}")
        #     # Handle the error as needed
        #     raise requests.exceptions.HTTPError(f"Status {response.status_code}: {response.text}")        
        # response.raise_for_status()
        # chat_completion_result = response.json()

        # chat_completion_result = chat_completion_result["choices"]
        # chat_completion_result = chat_completion_result[0]
        # chat_completion_result = chat_completion_result["message"]
        # chat_completion_result = chat_completion_result["content"]

        # print (chat_completion_result)
        self.messages.append({"role": "assistant", "content": response})
        return response

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

# Function to call the LLM API  
def call_llm_api(coder, tools=None):  
    """Call the LLM API directly"""  
    # url = "https://llm-api.amd.com/openai/deployments/swe-gpt4o-exp1/chat/completions"  
    # api_key = os.environ["API_KEY"]  
    # headers = {"Ocp-Apim-Subscription-Key": api_key}  
      
    body = {  
        "messages": coder.messages,  
        "temperature": coder.temperature,  
        "top_p": coder.top_p,  
    }  
      
    if tools:  
        body["tools"] = tools  
    
    if coder.workflow == "byu":
        body["model"] = coder.model # The model is only needed for the byu API


    print(f"API Request: {json.dumps(body, indent=2)}")  
    body_json = json.dumps(body, indent=2)


    response = requests.post(url=coder.url, json=body, headers=coder.headers)  
      
    # Print the raw response for debugging  
    print(f"API Response status: {response.status_code}")  
    try:  
        response_json = response.json()  
        print(f"API Response: {json.dumps(response_json, indent=2)}")  
        return response_json  
    except json.JSONDecodeError:  
        print(f"Raw response: {response.text}")  
        raise ValueError(f"Failed to decode JSON from API response: {response.text}")  

# Convert function to OpenAI tool format with correct JSON Schema types  
def function_to_tool(func):  
    """Convert a Python function to OpenAI tool format"""  
    import inspect  
    sig = inspect.signature(func)  
    parameters = {}  
      
    for name, param in sig.parameters.items():  
        annotation = param.annotation  
        if annotation == inspect.Parameter.empty:  
            type_str = "string"  
        elif annotation == int or annotation == float:  
            # Use "number" instead of "float" or "int" for JSON Schema compatibility  
            type_str = "number"  
        else:  
            type_str = "string"  
              
        parameters[name] = {  
            "type": type_str,  
            "description": f"Parameter {name} for {func.__name__}"  
        }  
      
    return {  
        "type": "function",  
        "function": {  
            "name": func.__name__,  
            "description": func.__doc__ or f"Function {func.__name__}",  
            "parameters": {  
                "type": "object",  
                "properties": parameters,  
                "required": list(parameters.keys())  
            }  
        }  
    }  

def format_node_content(node):  
    content = node.get_content()  
    # Detect if content is code and preserve formatting  
    if content.strip().startswith(('def ', 'class ', '#include', '//')) or '{' in content:  
        return f"```cpp\n{content}\n```"  
    return content  

# Initialize the RAG components with appropriate embedding  
def setup_rag():  
    print("Loading documents...")  
    documents = SimpleDirectoryReader("./rag_data").load_data()  
    print("Creating embedding model...")  
      
    # Use HuggingFaceEmbedding if available, otherwise fallback to MockEmbedding  
    if HuggingFaceEmbedding is not None:  
        try:  
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  
            print("Using HuggingFaceEmbedding")  
        except Exception as e:  
            print(f"Error creating HuggingFaceEmbedding: {e}")  
            print("Falling back to MockEmbedding")  
            embed_model = MockEmbedding(embed_dim=1536)  
    else:  
        print("Using MockEmbedding")  
        embed_model = MockEmbedding(embed_dim=1536)  
      
    # Disable LLM in the global settings - we'll provide our own  
    Settings.llm = None  
    Settings.embed_model = embed_model  
      
    print("Creating vector index...")  
    index = VectorStoreIndex.from_documents(documents)  
    return index, embed_model  

# Create a RAG search function that directly queries the index  
def search_documents(coder, query: str, index) -> str:  
    """Useful for answering natural language questions."""  
    # Since we can't use the standard query_engine with our custom LLM,  
    # let's directly get the nodes and construct a response  
    # Adjust retriever to get most relevant k examples  
    retriever = index.as_retriever(similarity_top_k=3)  
    nodes = retriever.retrieve(query)  
      
    if not nodes:  
        return "No relevant information found."  
      
    # Construct the prompt with retrieved content  
    formatted_examples = []  
    for i, node in enumerate(nodes):  
        content = node.get_content()  
        example_name = node.metadata.get('file_name', f'Example {i+1}')  
        formatted_examples.append(f"### {example_name}\n```cpp\n{content}\n```")  

    context = "\n\n".join(formatted_examples)  


    prompt = f"""Write optimized HLS code using these examples as reference:  

    {context}  

    Your task: {query}  
    
    Follow the patterns in the examples, particularly the optimization techniques, pragma usage, and memory access patterns. Implement a complete, working solution.  
    """        
    # Call our API directly  
    coder.messages.append({"role": "user", "content": prompt})  
    response = call_llm_api(coder)  
      
    if "response" in response and "error" in response["response"]:  
        return f"Error: {response['response']['error']['message']}"  
      
    if "choices" in response and len(response["choices"]) > 0:  
        return response["choices"][0]["message"]["content"]  
    else:  
        return f"Error: Unable to get a proper response from the LLM API. Response: {response}"  

# RAG agent implementation  
def rag_agent(coder, index):  
    """A RAG agent implementation using direct API calls"""  
      
    # Create a wrapper for search_documents that includes the index  
    def search_wrapper(query: str) -> str:  
        return search_documents(coder, query, index)  
      
    # First, get the LLM to decide what to do  
    tools = [  
        function_to_tool(search_wrapper)  
    ]  
      
    print("\nStarting RAG agent interaction...")  
    response = call_llm_api(coder, tools=tools)  
      
    # Check if the response has the expected format  
    if "response" in response and "error" in response["response"]:  
        return f"Error: {response['response']['error']['message']}"  
      
    if "choices" not in response or len(response["choices"]) == 0:  
        return f"Error: Unable to get a proper response from the LLM API. Response: {response}"  
      
    assistant_message = response["choices"][0]["message"]  
    messages = coder.messages.copy()  # Start with the existing message history  
    messages.append(assistant_message)  # Add the assistant's response        

    # If the LLM wants to use a tool  
    if "tool_calls" in assistant_message:  
        print("\nLLM decided to use tools")  
        for tool_call in assistant_message["tool_calls"]:  
            function_name = tool_call["function"]["name"]  
            arguments = json.loads(tool_call["function"]["arguments"])  
              
            print(f"Executing tool: {function_name} with arguments: {arguments}")  
              
            # Execute the function   
            if function_name == "search_wrapper":  
                result = search_wrapper(**arguments)  
            else:  
                result = f"Unknown function: {function_name}"  
                  
            print(f"Tool result: {result}")  
                  
            # Add the function response to messages  
            messages.append({  
                "role": "tool",  
                "tool_call_id": tool_call["id"],  
                "content": str(result)  
            })  
          
        # Get the final response  
        print("\nGetting final response after tool calls...")  
        temp_coder = HLSCoder(  
            url=coder.url,  
            workflow=coder.workflow,  
            model=coder.model,  
            temperature=coder.temperature,  
            top_p=coder.top_p  
        )  
        temp_coder.messages = messages  
        final_response = call_llm_api(temp_coder)  
          
        if "response" in final_response and "error" in final_response["response"]:  
            return f"Error: {final_response['response']['error']['message']}"  
          
        if "choices" not in final_response or len(final_response["choices"]) == 0:  
            return f"Error: Unable to get a proper final response from the LLM API. Response: {final_response}"  
              
        return final_response["choices"][0]["message"]["content"]  
    else:  
        # LLM answered directly  
        print("\nLLM answered directly without using tools")  
        return assistant_message["content"]  