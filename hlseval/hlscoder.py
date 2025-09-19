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


from typing import Dict, Any, List  
from hlseval.splitter import MultiContentSplitter
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings  
from llama_index.core import MockEmbedding  
from llama_index.core import StorageContext, load_index_from_storage  

logger = logging.getLogger("HLSCODER")



# Try to import HuggingFaceEmbedding from different possible locations  
try:  
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding  
except ImportError:  
    HuggingFaceEmbedding = None  


api_key = os.environ["API_KEY"] # reads for the environment variable 'API_KEY'
openrouter_api_key = os.environ["OPENROUTER_API_KEY"] # reads for the environment variable 'OPENROUTER_API_KEY'


def count_tokens(text, model="gpt-4o"):
    """Estimate token count for a given text string."""
    try:
        encoder = tiktoken.encoding_for_model(model)
        return len(encoder.encode(text))
    except:
        # Fallback: rough estimate (4 chars ≈ 1 token)
        return len(text) // 4


def get_messages_token_count(messages, model="gpt-4o"):
    """Count tokens in a list of messages."""
    total_tokens = 0
    for message in messages:
        # Count the message role and content
        total_tokens += count_tokens(message.get("role", "") + message.get("content", ""), model)
        # Add a small overhead for message formatting
        total_tokens += 4  # Rough estimate of message formatting overhead
    return total_tokens

# Then update the manage_context_window method in HLSCoder


class HLSCoder:
    """A basic agent that talks via OpenAI APIs"""
    def __init__(  
        self, hls_flow, system_prompt, 
        url="https://openrouter.ai/api/v1/chat/completions", model="gpt-4o", temperature=0.0, top_p=1.0, attempts=1  
    ):  
        self.system_prompt=system_prompt  
        self.messages = [{"role": "system", "content": self.system_prompt}]  
        self.model = model  
        self.temperature = temperature  
        self.top_p = top_p  
        self.attempts = attempts  
        self.hls_flow = hls_flow
        self.token_usage = {  
            "completion_tokens": 0,  
            "prompt_tokens": 0,  
            "total_tokens": 0,  
        }
        self.url=url  
        self.headers = {"Authorization": f"Bearer {openrouter_api_key}"}


    def __call__(self, prompt):
        """An HLSCoder instance can be called with a user prompt. If self.attempts = 1 it will do a one-shot
        response based on the prompt, if self.attempts > 1 it will check if the generated output compiles and
        retry self.attempts number of times.
        """
        self.prompt = prompt
        self.messages.append({"role": "user", "content": prompt})

        response = self.generate_code()
        return {
            "response": response,
            # "attempt": 0,
            "token_usage": self.token_usage,
            "history": self.messages,
        }

    def process_response(self, api_response):
        """Process the API response, handling different formats"""
        
        try:
            # Check if this is a Claude-style response
            if "content" in api_response and isinstance(api_response["content"], list):
                # Claude format
                content_items = api_response["content"]
                response_text = ""
                for item in content_items:
                    if item.get("type") == "text":
                        response_text += item["text"]
                response = response_text
            
            # Check if this is an OpenAI-style response (Llama, DeepSeek)
            elif "choices" in api_response and len(api_response["choices"]) > 0:
                if "message" in api_response["choices"][0]:
                    response = api_response["choices"][0]["message"]["content"]
                else:
                    response = api_response["choices"][0].get("text", "")
            
            # Check if this is a Gemini-style response
            elif "candidates" in api_response and len(api_response["candidates"]) > 0:
                candidate = api_response["candidates"][0]
                if "content" in candidate and "parts" in candidate["content"]:
                    # Extract text from all parts and combine
                    parts = candidate["content"]["parts"]
                    response = "".join(part.get("text", "") for part in parts)
                else:
                    # Fallback if structure is different
                    response = str(candidate)
            
            else:
                # Fallback - try to extract any text we can find
                logger.warning("HLS CODER: Unknown response format")
                response = str(api_response)
            
            logger.debug("HLS CODER: Final Response:")  
            logger.debug("HLS CODER: " + response)  
            
            self.messages.append({"role": "assistant", "content": response})  
            return response
        except Exception as e:
            logger.error(f"HLS CODER: Error processing response: {str(e)}")
            return f"Error processing LLM response: {str(e)}"

    def manage_context_window(self, max_tokens=6000):
        """Aggressively manage context window to prevent token limit errors."""  
        # Always keep the system prompt  
        system_message = self.messages[0]  
        
        # Count tokens in current messages  
        total_tokens = get_messages_token_count(self.messages, self.model)
        logger.debug(f"Current context size: {total_tokens} tokens")  
        
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
        new_tokens = get_messages_token_count(self.messages, self.model)
        logger.debug(f"Reduced context size from {total_tokens} to {new_tokens} tokens")  

    def generate_code(self):
        """Lightweight wrapper to generate a response using OpenAI API with strict context management."""  
        # First try aggressive context window management  
        self.manage_context_window(max_tokens=6000)  # Adjust based on your model's limits

        
        # Before calling the API, verify we're within limits  
        token_count = get_messages_token_count(self.messages)
        if token_count > 7000:  # Safety margin  
            logger.warning(f"Context size {token_count} may exceed limits. Applying emergency reduction.")  
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
            response = self.process_response(api_response)
            return response  
        except Exception as e:  
            logger.error(f"Error calling LLM API: {e}")  
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
            r"```(?:[a-zA-Z0-9#+]+)?\n(.*?)```|```(.*?)```", text, re.DOTALL
        )
        # Flatten the list and filter out empty matches
        code_blocks = [block for match in code_blocks for block in match if block]
        return code_blocks[0].strip() if code_blocks else None


# Function to call the LLM API  
def call_llm_api(coder, tools=None):  
    """Call the LLM API directly"""
    body = {  
        "messages": coder.messages,  
        "temperature": coder.temperature,  
        "top_p": coder.top_p,  
    }
    if tools:  
        body["tools"] = tools
    if "claude" in coder.url.lower() or "gemini" in coder.url.lower(): # The Claude models needed a max_tokens value
        body["max_tokens"] = 8000
    logger.debug(f"HLS CODER: API Request body: {json.dumps(body, indent=2)}")
    data = json.dumps({
        "model": coder.model, 
        "messages": coder.messages
    })
    response = requests.post(url=coder.url, data=data, headers=coder.headers)
    # Log the raw response for debugging  
    logger.debug(f"API Response status: {response.status_code}")  
    try:  
        response_json = response.json()  
        logger.debug(f"API Response: {json.dumps(response_json, indent=2)}")  
        return response_json  
    except json.JSONDecodeError:  
        logger.debug(f"Raw response: {response.text}")  
        raise ValueError(f"Failed to decode JSON from API response: {response.text}")  

