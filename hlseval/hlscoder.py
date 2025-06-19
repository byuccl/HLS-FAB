import logging
import openai
import subprocess
import tempfile
import re
import requests
import os
import json  
import inspect  
import tiktoken
import hlseval 


from typing import Dict, Any, List  
from hlseval.splitter import MultiContentSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings  
from llama_index.core import MockEmbedding  
from llama_index.core import StorageContext, load_index_from_storage  

logger = logging.getLogger("__main__")



# Try to import HuggingFaceEmbedding from different possible locations  
try:  
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  
except ImportError:  
    try:  
        from llama_index_embeddings_huggingface import HuggingFaceEmbedding  
    except ImportError:  
        HuggingFaceEmbedding = None  

# Python Prompt
# SYSTEM_PROMPT = """You are a part of a code generation system for python.
# Your job is to write Python code for a single kernel that will be run on a standard CPU..
# Produce only the python code for the requested kernel including any required headers and imports.
# Make sure the Python code is complete and self contained in a single code block.
# Produce only the kernel function, no main, no additional examples or code.
# """

_RAG_INDEX = None  
_EMBED_MODEL = None 

def get_or_create_rag_components():  
    """Singleton-like function to get or create RAG components"""  
    global _RAG_INDEX, _EMBED_MODEL  
      
    if _RAG_INDEX is None:  
        logger.info("HLS CODER: Creating new RAG index...")  
          
        # Check if we have a saved index  
        if os.path.exists("storage"):  
            try:  
                logger.debug("HLS CODER: Loading index from disk...")  
                # Set up the embedding model  
                if HuggingFaceEmbedding is not None:    
                    try:    
                        _EMBED_MODEL = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")    
                        logger.debug("HLS CODER: Using HuggingFaceEmbedding")    
                    except Exception as e:    
                        logger.warning(f"HLS CODER: Problem creating HuggingFaceEmbedding: {e}")    
                        logger.warning("HLS CODER: Falling back to MockEmbedding")    
                        _EMBED_MODEL = MockEmbedding(embed_dim=1536)    
                else:    
                    logger.warning("HLS CODER: Using MockEmbedding")    
                    _EMBED_MODEL = MockEmbedding(embed_dim=1536)  
                  
                # Disable LLM in the global settings - we'll provide our own    
                Settings.llm = None    
                Settings.embed_model = _EMBED_MODEL  
                  
                # Load the saved index  
                storage_context = StorageContext.from_defaults(persist_dir="storage")  
                _RAG_INDEX = load_index_from_storage(storage_context)  
                logger.debug("HLS CODER: Successfully loaded index from disk")  
            except Exception as e:  
                logger.warning(f"HLS CODER: Problem loading index from disk: {e}")  
                logger.warning("HLS CODER: Creating new index...")  
                _RAG_INDEX, _EMBED_MODEL = setup_rag()  
                # Save the index for future use  
                _RAG_INDEX.storage_context.persist("storage")  
        else:  
            logger.warning("HLS CODER: No saved index found, creating new index...")  
            _RAG_INDEX, _EMBED_MODEL = setup_rag()  
            # Save the index for future use  
            _RAG_INDEX.storage_context.persist("storage")  
      
    return _RAG_INDEX, _EMBED_MODEL  


# HLS C++ prompt
SYSTEM_PROMPT = """You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Vitis HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Make this code so that it is Deterministic when synthesized.
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
        
        # Get RAG components from the singleton function if RAG is enabled  
        if self.rag:  
            self.index, self.embed_model = get_or_create_rag_components()  
        else:  
            self.index = None  
            self.embed_model = None  
    
        self.url=url  
        if self.workflow == "byu":  
            self.headers = byu_headers   
        elif self.workflow == "amd":  
            self.headers = amd_headers  


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

    def manage_context_window(self, max_messages=10, max_tokens=64000):  
        """Keep only the most recent and relevant messages to stay within token limits."""  
        # Always keep the system prompt  
        system_message = self.messages[0]  
        
        # Count current messages (excluding system message)  
        current_messages = self.messages[1:]  
        
        # If we have more than max_messages, trim the oldest ones  
        # But preserve user/assistant pairs (question and answer)  
        if len(current_messages) > max_messages:  
            # Keep the most recent messages, preserving pairs  
            # Start with an even number to keep complete pairs  
            start_idx = max(1, len(current_messages) - max_messages)  
            if start_idx % 2 == 1:  # Make sure we start with a user message  
                start_idx -= 1  
                
            # Rebuild the message list with system prompt + recent messages  
            self.messages = [system_message] + current_messages[start_idx:]  
            
            # Add a context note at the beginning to inform the model  
            self.messages.insert(1, {  
                "role": "system",   
                "content": f"Note: {start_idx} earlier messages have been removed to stay within context limits."  
            })  

    def get_focused_rag_examples(self, query):  
        """Get highly condensed code examples from RAG to fit in context window."""  
        if not self.index:  
            return ""  
            
        # Use fewer examples when context is tight  
        retriever = self.index.as_retriever(similarity_top_k=2)  
        nodes = retriever.retrieve(query)  
        
        if not nodes:  
            return ""  
        
        # Extract only the most relevant code sections  
        patterns = []  
                
        # Look for function name/signature in the query  
        function_pattern = re.search(r'(void|int|float|double|bool)\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', query)  
        function_name = function_pattern.group(2) if function_pattern else None  
        
        # Look for key algorithmic terms  
        key_terms = re.findall(r'\b(sort|search|matrix|vector|array|loop|unroll|pipeline)\w*\b', query.lower())  
        
        for node in nodes:  
            content = node.get_content()  
            
            # Get file name for reference  
            example_name = node.metadata.get('file_name', 'example')  
            
            # Apply more aggressive filtering based on query  
            relevant_sections = []  
            
            # If we identified a function name in the query, prioritize matching functions  
            if function_name and function_name in content:  
                # Try to extract just the relevant function  
                func_match = re.search(rf'(void|int|float|double|bool)\s+{function_name}\s*\([^{{]*{{[^}}]*}}', content, re.DOTALL)  
                if func_match:  
                    relevant_sections.append(f"// Relevant function:\n{func_match.group(0)}")  
            
            # Extract only the most relevant pragmas for the task  
            pragmas = re.findall(r'#pragma\s+HLS[^\n]*', content)  
            if pragmas:  
                # Keep only a limited number of unique pragmas  
                unique_pragmas = set(pragmas)  
                if len(unique_pragmas) > 5:  # Limit to 5 pragmas max  
                    relevant_sections.append("// Key pragmas:\n" + "\n".join(list(unique_pragmas)[:5]))  
                else:  
                    relevant_sections.append("// Key pragmas:\n" + "\n".join(unique_pragmas))  
            
            # Look for code patterns related to key terms  
            for term in key_terms:  
                # Find lines containing the term  
                for line in content.split('\n'):  
                    if term in line.lower() and len(line) < 100:  # Avoid huge lines  
                        relevant_sections.append(f"// {term} pattern: {line.strip()}")  
            
            # If we found relevant sections, add them  
            if relevant_sections:  
                condensed = f"From {example_name}:\n" + "\n".join(relevant_sections)  
                patterns.append(condensed)  
        
        # Return a very condensed version, or nothing if we couldn't find relevant code  
        if patterns:  
            return "RELEVANT CODE PATTERNS:\n" + "\n\n".join(patterns)  
        else:  
            return "No directly relevant code patterns found."  

    def count_tokens(self, text, model="gpt-4o"):  
        """Estimate token count for a given text string."""  
        try:  
            encoder = tiktoken.encoding_for_model(model)  
            return len(encoder.encode(text))  
        except:  
            # Fallback: rough estimate (4 chars ≈ 1 token)  
            return len(text) // 4  
  
    def get_messages_token_count(self, messages, model="gpt-4o"):  
        """Count tokens in a list of messages."""  
        total_tokens = 0  
        for message in messages:  
            # Count the message role and content  
            total_tokens += self.count_tokens(message.get("role", "") + message.get("content", ""), model)  
            # Add a small overhead for message formatting  
            total_tokens += 4  # Rough estimate of message formatting overhead  
        return total_tokens  
  
    # Then update the manage_context_window method in HLSCoder  
    def manage_context_window(self, max_tokens=6000):  
        """Aggressively manage context window to prevent token limit errors."""  
        # Always keep the system prompt  
        system_message = self.messages[0]  
        
        # Count tokens in current messages  
        total_tokens = self.get_messages_token_count(self.messages, self.model)  
        logger.debug(f"HLS CODER: Current context size: {total_tokens} tokens")  
        
        # If we're under the limit, no need to truncate  
        if total_tokens <= max_tokens:  
            return  
        
        # Start with the essential messages  
        essential_messages = [system_message]  
        
        # Always keep the most recent user query  
        last_user_message = None  
        for i in range(len(self.messages)-1, 0, -1):  
            if self.messages[i]["role"] == "user":  
                last_user_message = self.messages[i]  
                break  
        
        # If we have RAG and the last message contains RAG results, simplify it  
        if self.rag and last_user_message and "Based on these relevant examples:" in last_user_message["content"]:  
            # Extract just the original query  
            parts = last_user_message["content"].split("Please implement the following:")  
            if len(parts) > 1:  
                # Keep only the original query part, discard the RAG examples  
                last_user_message["content"] = parts[1].strip()  
        
        # Add a note about context reduction  
        essential_messages.append({  
            "role": "system",  
            "content": "Note: Previous conversation history has been removed to stay within token limits."  
        })  
        
        # Add the current query  
        if last_user_message:  
            essential_messages.append(last_user_message)  
        
        # Replace the messages with the essential ones  
        self.messages = essential_messages  
        
        # Log the new token count  
        new_tokens = self.get_messages_token_count(self.messages, self.model)  
        logger.debug(f"HLS CODER: Reduced context size from {total_tokens} to {new_tokens} tokens")  

    def generate_code(self):  
        """Lightweight wrapper to generate a response using OpenAI API with strict context management."""  
        # First try aggressive context window management  
        self.manage_context_window(max_tokens=6000)  # Adjust based on your model's limits  
        
        if self.rag:  
            try:  
                # Get the last user message  
                current_prompt = self.messages[-1]["content"]  
                
                # Extract only essential requirements from the prompt  
                # This helps reduce the size of the RAG query  
                essential_prompt = current_prompt  
                if len(current_prompt) > 500:  
                    # Extract just the key parts (function signature, requirements)  
                    function_sig = re.search(r'(void|int|float|double|bool)\s+\w+\s*\([^)]*\)', current_prompt)  
                    if function_sig:  
                        essential_prompt = function_sig.group(0) + "\n" + current_prompt[-500:]  
                
                # Get very focused examples  
                focused_examples = self.get_focused_rag_examples(essential_prompt)  
                
                # Create a minimal prompt with just the core requirements  
                focused_prompt = f"TASK: {current_prompt}\n\nRELEVANT EXAMPLES:\n{focused_examples}"  
                
                # Replace the last message with the focused prompt  
                self.messages[-1]["content"] = focused_prompt  
                
                # Check if we're still within limits  
                if self.get_messages_token_count(self.messages) > 6500:  
                    # Further reduce by eliminating all but the last message  
                    self.messages = [  
                        {"role": "system", "content": self.system_prompt},  
                        {"role": "user", "content": focused_prompt}  
                    ]  
            except Exception as e:  
                logger.error(f"HLS CODER: Error during RAG processing: {e}")  
                # Fall back to basic prompt if RAG fails  
        
        # Before calling the API, verify we're within limits  
        token_count = self.get_messages_token_count(self.messages)  
        if token_count > 7000:  # Safety margin  
            logger.warning(f"HLS CODER: Context size {token_count} may exceed limits. Applying emergency reduction.")  
            # Emergency reduction - just keep system prompt and a simplified version of the last query  
            last_query = self.messages[-1]["content"]  
            if len(last_query) > 1500:  
                last_query = last_query[:1500] + "...[truncated]"  
            
            self.messages = [  
                {"role": "system", "content": self.system_prompt},  
                {"role": "user", "content": last_query}  
            ]  
        
        # Call the LLM API with the optimized context  
        try:  
            api_response = call_llm_api(self)  
            response = api_response["choices"][0]["message"]["content"]  
            
            logger.debug("HLS CODER: Final Response:")  
            logger.debug("HLS CODER: " + response)  
            
            self.messages.append({"role": "assistant", "content": response})  
            return response  
        except Exception as e:  
            logger.error(f"HLS CODER: Error calling LLM API: {e}")  
            # Return an error message that can be displayed to the user  
            error_response = f"Error generating code: {str(e)}"  
            self.messages.append({"role": "assistant", "content": error_response})  
            return error_response  

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


    logger.debug(f"HLS CODER: API Request: {json.dumps(body, indent=2)}")  

    response = requests.post(url=coder.url, json=body, headers=coder.headers)  
      
    # Log the raw response for debugging  
    logger.debug(f"HLS CODER: API Response status: {response.status_code}")  
    try:  
        response_json = response.json()  
        logger.debug(f"HLS CODER: API Response: {json.dumps(response_json, indent=2)}")  
        return response_json  
    except json.JSONDecodeError:  
        logger.debug(f"HLS CODER: Raw response: {response.text}")  
        raise ValueError(f"Failed to decode JSON from API response: {response.text}")  

# Convert function to OpenAI tool format with correct JSON Schema types  
def function_to_tool(func):  
    """Convert a Python function to OpenAI tool format"""  
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
    """Initialize RAG with a multi-content splitter for diverse document types."""  
    logger.debug("HLS CODER: Loading documents...")  
    documents = SimpleDirectoryReader("./rag_data").load_data()  
      
    logger.debug("HLS CODER: Creating multi-content splitter...")  
    content_splitter = hlseval.MultiContentSplitter()  
      
    logger.debug("HLS CODER: Creating embedding model...")  
    if HuggingFaceEmbedding is not None:  
        try:  
            embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")  
            logger.debug("HLS CODER: Using HuggingFaceEmbedding")  
        except Exception as e:  
            logger.warning(f"HLS CODER: Error creating HuggingFaceEmbedding: {e}")  
            logger.warning("HLS CODER: Falling back to MockEmbedding")  
            embed_model = MockEmbedding(embed_dim=1536)  
    else:  
        logger.warning("HLS CODER: Using MockEmbedding")  
        embed_model = MockEmbedding(embed_dim=1536)  
      
    # Configure settings  
    Settings.llm = None  
    Settings.embed_model = embed_model  
      
    logger.debug("HLS CODER: Splitting documents based on content type...")  
    # Split documents into nodes  
    nodes = content_splitter.split_documents(documents)  
      
    logger.debug(f"HLS CODER: Created {len(nodes)} nodes from {len(documents)} documents")  
      
    logger.debug("HLS CODER: Creating vector index...")  
    # Create index from nodes instead of documents  
    index = VectorStoreIndex(nodes)  
      
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
      
    logger.info("HLS CODER: Starting RAG agent interaction...")  
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
        logger.info("HLS CODER: LLM decided to use tools")  
        for tool_call in assistant_message["tool_calls"]:  
            function_name = tool_call["function"]["name"]  
            arguments = json.loads(tool_call["function"]["arguments"])  
              
            logger.info(f"HLS CODER: Executing tool: {function_name} with arguments: {arguments}")  
              
            # Execute the function   
            if function_name == "search_wrapper":  
                result = search_wrapper(**arguments)  
            else:  
                result = f"Unknown function: {function_name}"  
                  
            logger.debug(f"HLS CODER: Tool result: {result}")  
                  
            # Add the function response to messages  
            messages.append({  
                "role": "tool",  
                "tool_call_id": tool_call["id"],  
                "content": str(result)  
            })  
          
        # Get the final response  
        logger.debug(" HLS CODER: Getting final response after tool calls...")  
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
        logger.info("HLS CODER: LLM answered directly without using tools")  
        return assistant_message["content"]  
