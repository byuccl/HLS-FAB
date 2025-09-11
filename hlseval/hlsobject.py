from datetime import datetime
import logging
import pathlib
import json
import os
import re
import hlseval.utils as utils
from hlseval.hlscoder import HLSCoder
import hlseval.prompts as prompts
import hlseval.parser as parser
from .vitis import CsimTimeoutError, CsynthTimeoutError, run_vitis_tools

logger = logging.getLogger(__name__)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

class HLSEval(HLSCoder):
    def __init__(self,
            model: str="openai/gpt-4.1-mini",
            kernel: dict=None,
            hls_tool_chain: str="vitis",
            iteration: int = 1,
            num_attempts: int = 1,
            skip_behavioral: bool = False,
            rapidwright: bool = False,
            goals = None):
        self._model = model
        self._kernel = kernel
        self._hls_tool = hls_tool_chain
        self._iter = iteration
        self._num_attempts = num_attempts
        self._skip_behavioral = skip_behavioral
        self._rapidwright = rapidwright
        self._goals = goals
        self._kernel_name = kernel["kernel_name"]
        self._logfile = f"results/solutions/{hls_tool_chain}/{model}/{iteration+1}/logs/timing.log"

        self._solution_path = f"./results/solutions/{hls_tool_chain}/{model}/{self._iter+1}/{self._kernel_name}"
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

    def llm_call(self, attempt: int, feedback=None):


        logger.debug(f"Generating LLM Call for {self.kernel_name} attempt {attempt+1}")
        # Generate the Prompt for LLM
        if attempt > 0:
            prev_solutions_path = f"{self.solution_path}/{attempt}"
            with open(f"{prev_solutions_path}/{self.kernel_name}.cpp", "r") as f:
                prev_code = f.read()
            prompt = prompts.feedback_prompt_vitis(prev_code=prev_code, feedback=feedback) if self.hls_flow == "vitis" else prompts.feedback_prompt_dyarapid(prev_code=prev_code, feedback=feedback)
            logger.debug(prompt)
        else: 
            prompt = self._kernel["prompt"]


        url = OPENROUTER_URL
        
        llm_completion_begin = datetime.now()

        response = self(prompt)
        result = {
            "code": self.extract_codeblock(response["response"]),
            "stats": {
                "token_usage": response["token_usage"],
                "history": response["history"],
            },
        }
        new_solution_path = f"{self.solution_path}/{attempt+1}"
        generated_code = result["code"]

        llm_completion_end = datetime.now()
        llm_elapsed_time = llm_completion_end - llm_completion_begin
        logger.debug(f"LLM API call elapsed time: {llm_elapsed_time}")

        with open(f"{new_solution_path}/{self.kernel_name}.json", "w") as file:
            json.dump(generated_code, file, indent=4)

        with open(f"{new_solution_path}/{self.kernel_name}.cpp", "w") as file:
            logger.debug("Writing to file")
            kernel_code = self.get_kernel_code(new_solution_path)
            if kernel_code == None:
                    logger.warning("LLM Failed to produce Code. Skipping attempt")
                    return 1, "Failed to produce kernel code"
            else:
                file.write(kernel_code)

        return llm_elapsed_time

    # Function for running all the tools and gathering the feedback
    def vitis_run_w_feedback(self, attempt: int, goals: str):
        vitis_feedback = {}
        new_solution_path = f"{self.solution_path}/{attempt+1}"
        if self._skip_behavioral:
            logger.info("Skipping CSIM")
            sim_returncode = 0 # report that csim was successful
        else:
            # Run CSIM    
            try:
                logger.info("Starting Vitis Run")
                csim_output, csim_elapsed_time = run_vitis_tools(new_solution_path, self.kernel_name, mode="csim") # FIXME: Switch to run_csim function
                sim_returncode = csim_output.returncode if (csim_output != None) else 1
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
                synth_output, csynth_elapsed_time = run_vitis_tools(new_solution_path, self.kernel_name, mode="csynth") # FIXME: Rename function to run_csynth
                vitis_feedback = parser.gather_parsed_data(self.kernel_name, self.hls_flow, self._model, attempt=attempt+1, goals=goals, iter=self._iter)
                synth_returncode = synth_output.returncode 

                logger.debug(f"CYNTH elapsed time: {csynth_elapsed_time}")  
            except CsynthTimeoutError as e:
                logger.warning(" Vitis timed out")
                vitis_feedback = parser.gather_parsed_data(self.kernel_name, self.hls_flow, self._model, attempt=attempt+1, goals=goals, iter=self._iter)
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
            vitis_feedback = parser.gather_parsed_data(self.kernel_name, self.hls_flow, self._model, attempt=attempt+1, goals=goals, iter=self._iter)
            vitis_feedback['sim_feedback']['errors'].extend(["Failed to complete Csim. Csim did not run successfully."])
            vitis_feedback['synth_feedback']['errors'].extend(["Synthesis not run due to failed C Simulation."])
            synth_returncode = 1   

        logger.info("--- Vitis Run Completed ---")

        out_data_path = f"{new_solution_path}/{self.kernel_name}_data.json"
        parser.save_feedback_data(out_data_path, vitis_feedback)
        

        returncode = (sim_returncode and synth_returncode) if sim_returncode else synth_returncode
        return returncode, vitis_feedback, csim_elapsed_time, csynth_elapsed_time

    # def run_kernel_attempt(
    #     self,
    #     attempt: int, 
    #     prev_feedback: dict=None, 
    #     goals: str=None
    #     ) -> tuple :

    #     llm_elapsed_time = None
    #     dynamatic_elapsed_time = None
    #     csim_elapsed_time = None
    #     csynth_elapsed_time = None
    #     new_solution_path = f"{self.solution_path}/{attempt+1}"


    #     llm_elapsed_time = self.llm_call(attempt, feedback=prev_feedback)


    #     if self.hls_flow == "vitis":
    #         returncode, feedback, csim_elapsed_time, csynth_elapsed_time = self.vitis_run_w_feedback(attempt, goals)

    #     elif self.hls_flow == "dynarapid":
    #         returncode, feedback, csim_elapsed_time, dynamatic_elapsed_time = run_dynarapid_w_feedback(self, attempt, goals)
   
    #     utils.add_table_row(
    #         self._logfile,
    #         self.kernel_name, 
    #         attempt,
    #         llm_elapsed_time, 
    #         dynamatic_elapsed_time, 
    #         csim_elapsed_time, 
    #         csynth_elapsed_time
    #     )
    #     return returncode, feedback

    # def process_kernel(self):
    #     prev_ret_code = -1
    #     testbench_status = "failure"
    #     testbench_match_status = ""
    #     feedback = None
    #     timestamp = datetime.now().strftime("%H:%M:%S")
    #     logger.debug(f"Kernel Start time: {timestamp}")

    #     for attempt in self._num_attempts:
    #         logger.info(f"Kernel: {self._kernel_name} Attempt: {attempt+1}")

    #         # dir_path = pathlib.Path(f"./results/solutions/{hls_flow}/{model}/{iter+1}/{kernel_name}/attempts_{attempt+1}")
    #         prev_ret_code, feedback = utils.run_kernel_attempt(self, attempt, feedback, goals=self._goals)
    #         logger.debug(feedback)

    #         if prev_ret_code == 0 and testbench_status == "success" and testbench_match_status == "match":
    #             logger.debug(f"{self._kernel_name} compiled successfully on attempt {attempt+1}.")

    #         timestamp = datetime.now().strftime("%H:%M:%S")
    #         logger.debug(f"[{self._model}] Attempt {attempt+1} for kernel {self._kernel_name} ended at {timestamp}")

    #     logger.info(f"All attempts completed for {self._kernel_name}")


    # Extract the code from the LLM response
    def get_kernel_code(self, new_solutions_path: str = None) -> str:
        """Fetch the kernel code from the provided solution path, if none provided default 
        to canonical solution."""
        if not new_solutions_path:
            return self.get_prompt + self.get_canonical # FIXME: I'm pretty sure this wouldn't work anymore. We probably need a try/catch style solution for no solution file
        
        with open(os.path.join(new_solutions_path, f"{self.kernel_name}.json"), 'r') as sol_file:
            solution = json.load(sol_file)
            if not solution.get('code'):
                if not solution.get('content'):
                    logger.warning(f"UTILS: No code available in {new_solutions_path} for {self.kernel_name}")
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