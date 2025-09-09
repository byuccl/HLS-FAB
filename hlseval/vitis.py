import subprocess
from datetime import datetime
import logging
from .csim import run_csim 

logger = logging.getLogger(__name__)


class CsynthTimeoutError(Exception):
    """Raised when Vitis HLS times out."""
    pass

class CsimTimeoutError(Exception):
    """Raised when Vitis HLS times out."""
    pass


# # Function for running all the tools and gathering the feedback
# def vitis_run_w_feedback(test: HLSEval, attempt: int, goals: str):
#     vitis_feedback = {}
#     new_solution_path = f"{test.solution_path}/{attempt+1}"
#     if test._skip_behavioral:
#         logger.info("Skipping CSIM")
#         sim_returncode = 0 # report that csim was successful
#     else:
#         # Run CSIM    
#         try:
#             logger.info("Starting Vitis Run")
#             csim_output, csim_elapsed_time = run_vitis_tools(new_solution_path, test.kernel_name, mode="csim")
#             sim_returncode = csim_output.returncode if (csim_output != None) else 1
#         except CsimTimeoutError as e:
#             sim_returncode = 1
#             logger.warning("Vitis timed out on Simulation")
#         except Exception as e:
#             sim_returncode = 1
#             logger.warning(f"An error occurred: {e}") 

#         logger.debug(f"CSIM elapsed time: {csim_elapsed_time}")

#     if sim_returncode == 0:
#     # Run Synthesis only if CSIM was successful
#         try:
#             logger.info("--- Running CSYNTH ---")
#             synth_output, csynth_elapsed_time = run_vitis_tools(new_solution_path, test.kernel_name, mode="csynth")
#             vitis_feedback = parser.gather_parsed_data(test.kernel_name, test.hls_flow, test._model, attempt=attempt+1, goals=goals, iter=test._iter)
#             synth_returncode = synth_output.returncode 

#             logger.debug(f"CYNTH elapsed time: {csynth_elapsed_time}")  
#         except CsynthTimeoutError as e:
#             logger.warning(" Vitis timed out")
#             vitis_feedback = parser.gather_parsed_data(test.kernel_name, test.hls_flow, test._model, attempt=attempt+1, goals=goals, iter=test._iter)
#             logger.debug(vitis_feedback)
#             logger.debug("vitis Feedback before Adding new Errors")
#             vitis_feedback['synth_feedback']['errors'].extend(["Failed to complete Synthesis. Synthesis took too long due to exploding instructions and resources utilization"])
#             vitis_feedback['synth_feedback']['goals'].extend([
#                 "Try pipelining inside of loops rather than outside to ensure loop pipelining not function pipelining."
#             ])
#             logger.debug(vitis_feedback)
#             logger.debug("vitis Feedback before Adding new Errors")
#             synth_returncode = 1
#     else:
#         vitis_feedback = parser.gather_parsed_data(test.kernel_name, test.hls_flow, test._model, attempt=attempt+1, goals=goals, iter=test._iter)
#         vitis_feedback['sim_feedback']['errors'].extend(["Failed to complete Csim. Csim did not run successfully."])
#         vitis_feedback['synth_feedback']['errors'].extend(["Synthesis not run due to failed C Simulation."])
#         synth_returncode = 1   

#     logger.info("--- Vitis Run Completed ---")

#     out_data_path = f"{new_solution_path}/{test.kernel_name}_data.json"
#     parser.save_feedback_data(out_data_path, vitis_feedback)
    

#     returncode = (sim_returncode and synth_returncode) if sim_returncode else synth_returncode
#     return returncode, vitis_feedback, csim_elapsed_time, csynth_elapsed_time


# Run vitis tools on the generated code
def run_vitis_tools(filepath, kernel_name, mode="csynth", hls_flow="vitis"):
    csynth_elapse_time = None

    if mode == "csynth":
        csynth_begin = datetime.now()
        with open("vitis_config.cfg", "w") as file:
            file.write(
                f"""
            part=xcu280-fsvh2892-2L-e

            [hls]
            flow_target=vivado
            package.output.format=ip_catalog
            package.output.syn=false
            syn.top={kernel_name}
            syn.file={filepath}/{kernel_name}.cpp
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
                # v++ -c --mode hls --config vitis_config.cfg --work_dir ./results/solutions/gpt-4o/...
                timeout=120,

                capture_output=True, 
                text=True
            )
            csynth_end = datetime.now()
            csynth_elapse_time = csynth_end - csynth_begin
            return result, csynth_elapse_time
        except subprocess.TimeoutExpired:
            raise CsynthTimeoutError("Vitis tools stalled during HLS Synthesis.")

    elif mode == "csim":
        return run_csim(kernel_name, filepath, hls_flow=hls_flow)
        
