from datetime import datetime as dt
import logging
import pathlib
import json
import os
import re
from typing import Any

from hlseval.hlscoder import HLSCoder
import hlseval.prompts as prompts
import hlseval.parser as parser
from .vitis import CsynthTimeoutError, run_csynth
from .vitis import run_csim, CsimTimeoutError
from .prompts import SYSTEM_PROMPTS

logger = logging.getLogger("HLSOBJECT")


class HLSEval(HLSCoder):
    def __init__(self,
            model: str="openai/gpt-4.1-mini",
            kernel: dict=None,
            hls_tool_chain: str="vitis",
            iteration: int = 1,
            num_attempts: int = 1,
            skip_behavioral: bool = False,
            rapidwright: bool = False,
            logfile = None,
            rag=False):
        
        super().__init__(
            hls_flow=hls_tool_chain, 
            system_prompt=SYSTEM_PROMPTS[f"{hls_tool_chain}"][f"skip_behavioral_{skip_behavioral}"], 
            model=model, 
            temperature=0.0, 
            top_p=1.0, 
            attempts=1, 
        )

        self._model = model
        self._kernel = kernel
        self._hls_tool = hls_tool_chain
        self._iter = iteration
        self._num_attempts = num_attempts
        self._skip_behavioral = skip_behavioral
        self._rapidwright = rapidwright
        self._kernel_name = kernel["kernel_name"]
        self._logfile = logfile
        self._rag = rag

        if self._rag:
            self._solution_path = f"./results/solutions/{hls_tool_chain}/rag/{model}/{self._iter+1}/{self._kernel_name}"
        else:
            self._solution_path = f"./results/solutions/{hls_tool_chain}/no_rag/{model}/{self._iter+1}/{self._kernel_name}"
        pathlib.Path(self._solution_path).mkdir(parents=True, exist_ok=True)

    @property
    def model_name(self):
        return self._model
    
    @property
    def kernel_name(self):
        return self._kernel_name

    @property 
    def solution_path(self):
        return self._solution_path
    
    def get_prompt(self):
        return self._kernel["prompt"]

    def get_canonical(self):
        return self._kernel["canonical_solution"]

    def llm_call(self, attempt: int, rag, feedback=None):
        logger.debug(f"Generating LLM Call for {self.kernel_name} attempt {attempt+1}")
        # Generate the Prompt for LLM
        if attempt > 0:
            prev_solutions_path = f"{self.solution_path}/attempts_{attempt}"
            with open(f"{prev_solutions_path}/{self.kernel_name}.cpp", "r") as f:
                prev_code = f.read()
            prompt = prompts.feedback_prompt_vitis(prev_code=prev_code, feedback=feedback) if self.hls_flow == "vitis" else prompts.feedback_prompt_dyarapid(prev_code=prev_code, feedback=feedback)
            logger.debug(prompt)
        else: 
            prompt = self._kernel["prompt"]
        
        llm_completion_begin = dt.now()
        
        if rag["use_rag"]:
            rag["num_retrieved"] = [2] # change this to increase number of retrieved kernel examples
            from llama_index.core import (
                StorageContext,
                load_index_from_storage,
                VectorStoreIndex,
                SimpleDirectoryReader
            )
            logger.debug(f"importeed llama-index")
            rag["directory"] = "./rag/vector_database"
            # Set up directory
            if not os.path.exists(rag["directory"]):
                # Index new directory
                logger.debug("Indexing...")
                documents = SimpleDirectoryReader("rag_data/kernels", recursive=True).load_data()
                index = VectorStoreIndex.from_documents(documents)
                index.storage_context.persist(persist_dir=rag["directory"])
            else:
                # Load existing directory
                storage_context = StorageContext.from_defaults(persist_dir=rag["directory"])
                index = load_index_from_storage(storage_context)
            # Find the most similar nodes
            retriever = index.as_retriever(similarity_top_k=2)
            nodes = retriever.retrieve(prompt)

            context_string = "Reference related hls kernel:\n"
            for node in nodes:
                logger.debug(f"rag reference kernel: {node.metadata['file_name']}")
                context_string += node.node.text
                prompt = prompt + "\n" + context_string + "\n\n"


        response = self(prompt)
        
        result = {
            "code": self.extract_codeblock(response["response"]),
            "stats": {
                "token_usage": response["token_usage"],
                "history": response["history"],
            },
        }
        new_solution_path = f"{self.solution_path}/attempts_{attempt+1}"
        llm_completion_end = dt.now()
        llm_elapsed_time = llm_completion_end - llm_completion_begin
        logger.debug(f"LLM API call elapsed time: {llm_elapsed_time}")
        
        os.makedirs(new_solution_path, exist_ok=True) # Create directory if doesn't already exits
        with open(f"{new_solution_path}/{self.kernel_name}.json", "w") as file:
            json.dump(result, file, indent=4)

        with open(f"{new_solution_path}/{self.kernel_name}.cpp", "w") as file:
            logger.debug("Writing to file")
            kernel_code = self.get_kernel_code(new_solution_path)
            if kernel_code is None:
                    logger.error("LLM Failed to produce Code. Skipping attempt")
                    return 1, "Failed to produce kernel code"
            else:
                file.write(kernel_code)
        return llm_elapsed_time

    # Function for running all the tools and gathering the feedback
    def vitis_run_w_feedback(self, attempt: int):
        vitis_feedback = {}
        new_solution_path = f"{self.solution_path}/attempts_{attempt+1}"
        csim_elapsed_time = None
        csynth_elapsed_time = None
        if self._skip_behavioral:
            logger.info("Skipping CSIM")
            sim_returncode = 0 # report that csim was successful
        else:
            # Run CSIM    
            try:
                logger.info("Starting Vitis Run")
                csim_output, csim_elapsed_time = run_csim(new_solution_path, self.kernel_name)
                sim_returncode = csim_output.returncode if (csim_output is not None) else 1
            except CsimTimeoutError as e:
                sim_returncode = 1
                logger.warning("Vitis timed out on Simulation")
            except Exception as e:
                sim_returncode = 1
                logger.warning(f"An error occurred: {e}") 
            logger.debug(f"CSIM elapsed time: {csim_elapsed_time}")
        if sim_returncode == 0:
        # Run Synthesis only if CSIM was successful
            try:
                logger.info("--- Running CSYNTH ---")
                synth_output, csynth_elapsed_time = run_csynth(new_solution_path, self.kernel_name)
                vitis_feedback = parser.gather_parsed_data(self, attempt=attempt)
                synth_returncode = synth_output.returncode 

                logger.debug(f"CYNTH elapsed time: {csynth_elapsed_time}")  
            except CsynthTimeoutError as e:
                logger.warning(" Vitis timed out")
                vitis_feedback = parser.gather_parsed_data(self, attempt=attempt)
                logger.debug(vitis_feedback)
                logger.debug("vitis Feedback before Adding new Errors")
                vitis_feedback['synth_feedback']['errors'].extend(["Failed to complete Synthesis. Synthesis took too long due to exploding instructions and resources utilization"])
                vitis_feedback['synth_feedback']['goals'].extend([
                    "Try pipelining inside of loops rather than outside to ensure loop pipelining not function pipelining."
                ])
                synth_returncode = 1
        else:
            vitis_feedback = parser.gather_parsed_data(self, attempt=attempt)
            vitis_feedback['sim_feedback']['errors'].extend(["Failed to complete Csim. Csim did not run successfully."])
            vitis_feedback['synth_feedback']['errors'].extend(["Synthesis not run due to failed C Simulation."])
            synth_returncode = 1   
        logger.info("--- Vitis Run Completed ---")
        out_data_path = f"{new_solution_path}/{self.kernel_name}_data.json"
        parser.save_feedback_data(out_data_path, vitis_feedback)
        returncode = (sim_returncode and synth_returncode) if sim_returncode else synth_returncode
        return returncode, vitis_feedback, csim_elapsed_time, csynth_elapsed_time


    # Extract the code from the LLM response
    def get_kernel_code(self, new_solutions_path: str = None) -> str | None | Any:
        """Fetch the kernel code from the provided solution path, if none provided default 
        to canonical solution."""
        try:
            with open(os.path.join(new_solutions_path, f"{self.kernel_name}.json"), 'r') as sol_file:
                solution = json.load(sol_file)
                logger.debug(solution)
                if not solution.get('code'):
                    if not solution.get('content'):
                        logger.warning(f"No code available in {new_solutions_path} for {self.kernel_name}")
                        return None
                    srccode = solution['content']
                    return srccode


                srccode = solution['code']
                if self._hls_tool == "vitis":
                    # if gpt decides to be too helpful and adds a main()... remove it
                    srccode = re.sub(r'int\s+main\s*\([^)]*\)\s*{[^{}]*({[^{}]*}[^{}]*)*}', '', srccode, flags=re.DOTALL)
                    
                    # cppheaderparser will complain if we don't remove trailing comments
                    srccode = srccode.split('// extern "C"')[0]
                    
                return srccode
        except:
            logger.error("No Solution Path Provided. No Code extracted")
            return None