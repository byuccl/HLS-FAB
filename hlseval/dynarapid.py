import os
import subprocess
import logging
import time
import json
import glob
from datetime import datetime
import pathlib
import hlseval.hlsobject as hls
import hlseval.parser as parser
import hlseval.vitis as vitis

logger = logging.getLogger(__name__)

def run_dynarapid_w_feedback(kernel_object: hls.HLSEval, attempt: int=0):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    new_solution_path = f"{kernel_object.solution_path}/attempts_{attempt+1}"
    if kernel_object._skip_behavioral:
        print("Skipping Behavioral")
        pass
    else:
    # Run CSIM    
        print("Running CSIM for Dynarapid")
        try:
            logger.info("Starting Dynarapid Simulation")
            dsim_output, csim_elapsed_time = vitis.run_csim(new_solution_path, kernel_object.kernel_name, hls_flow=kernel_object.hls_flow)
            logger.debug(dsim_output)
        except vitis.CsimTimeoutError as e:
            logger.warning("Vitis timed out on Simulation")
        logger.debug(f"Dynarapid Simulation elapsed time: {csim_elapsed_time}")
        dynarapid_feedback = parser.gather_parsed_data(kernel_object.kernel_name, hls_flow=kernel_object.hls_flow, model=kernel_object._model, attempt=attempt+1, iter=kernel_object._iter)
    

    returncode, stdout, dynamatic_elapsed_time, rapidwright_elapsed_time = run_dynarapid(kernel_object, attempt)
    dynarapid_feedback['compile_feedback'] = {"returncode": returncode}


    dynarapid_feedback = check_dot_dcp(kernel_object, attempt, dynarapid_feedback)
    #Save Dynrapid Log in the same log as Sim
    search_pattern = os.path.join(new_solution_path, "dynamatic_*.log")
    matching_files = glob.glob(search_pattern)
    dynarapid_report = matching_files[-1]
    print(matching_files)
    print(dynarapid_report)

    log_file_dir = f"{new_solution_path}/synth_logs/"
    log_file = f"{log_file_dir}/{kernel_object.kernel_name}_synth{timestamp}.log"
    try:
        with open(dynarapid_report, 'r') as file:
            report = file.read()

            with open(log_file, 'a') as log:
                log.write("\n\n----------------- Dynarapid Log --------------------\n\n")
                log.write(report)
    except FileExistsError:
        logger.warning(f"{log_file} or {dynarapid_report} Does not exist")

    return returncode, dynarapid_feedback, csim_elapsed_time, dynamatic_elapsed_time


def run_dynarapid(kernel_object: hls.HLSEval, attempt: int=0):
    dynamatic_elapsed_time = None
    java_elapsed_time = None
    new_solution_path = f"{kernel_object.solution_path}/attempts_{attempt+1}"
    # Extract the relevant part of the path
    # Get the path from solutions_path that matches the container's structure
    if "/results/solutions/dynarapid/" in new_solution_path:
        # Extract the path starting from /results/...
        container_path = "/mount" + new_solution_path[new_solution_path.find("/results/solutions/dynarapid/"):]
    else:
        # Use the full path as fallback
        container_path = f"/mount{new_solution_path}"
    
    logger.debug(f"Host path: {new_solution_path}")
    logger.debug(f"Container path: {container_path}")
    
    # Create the TCL script content
    syn_tcl = f"""
set_project .
set_top_file {kernel_object.kernel_name}.cpp
synthesize -use-lsq=false -simple-buffers=true -verbose
#cdfg
write_hdl

exit
    """
    
    # Set proper umask before creating files
    # This ensures new files have appropriate permissions
    old_umask = os.umask(0)
    try:
        # Write the TCL script to a file with proper permissions from the start
        filename = f"{new_solution_path}/synthesis.tcl"
        try:
            with open(filename, "w") as file:
                file.write(syn_tcl)
            logger.debug(f"Successfully wrote to {filename}")
        except Exception as e:
            logger.error(f"An error occurred while writing TCL script: {e}")
            return 1, "No feedback at this time", dynamatic_elapsed_time, java_elapsed_time
        
        # Prepare the source directory with proper permissions
        try:
            os.makedirs(f"{new_solution_path}/src", mode=0o777, exist_ok=True)
            os.makedirs(f"{new_solution_path}/reports", mode=0o777, exist_ok=True)
            
            # Use the cp command with the --preserve=mode option to preserve mode but not ownership
            subprocess.run(f"cp {new_solution_path}/{kernel_object.kernel_name}.cpp {new_solution_path}/src/", 
                          shell=True, check=True, text=True)
            
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
            return 1, "No feedback at this time", dynamatic_elapsed_time, java_elapsed_time

        # Start the container if it's not already running
        try:
            # Capture and discard the output from docker start
            result = subprocess.run("docker start dynarapid_container", 
                            shell=True, check=True, text=True, 
                            capture_output=True)
            logger.info("Started dynarapid_container")
            
            # Verify the container is running
            container_check = subprocess.run("docker ps | grep dynarapid_container", 
                                           shell=True, text=True, capture_output=True)
            logger.debug(f"Container status: {container_check.stdout}")
            
            if not container_check.stdout.strip():
                logger.error("Error: Container not running properly")
                return 1, "Container not running", dynamatic_elapsed_time, java_elapsed_time
                
        except Exception as e:
            logger.error(f"Error with container operations: {e}")
            return 1, "Container operation failed", dynamatic_elapsed_time, java_elapsed_time
        
        # Execute commands separately with timeout protection
        try:           
            # Run the dynamatic command
            logger.info("Running dynamatic synthesis...")

            cmd = f"docker exec dynarapid_container bash -c 'cd {container_path} && dynamatic synthesis.tcl'"
            
            # Use Popen to have more control over the process
            dynamatic_begin = datetime.now()
            process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                    stderr=subprocess.PIPE, text=True)
            
            # Wait with timeout
            try:
                stdout, stderr = process.communicate(timeout=300)  # 5 minutes timeout
                dynamatic_end = datetime.now()
                logger.debug(f"Synthesis exit code: {process.returncode}")
                logger.debug(f"Synthesis output: {stdout}")
                logger.debug(f"Synthesis error: {stderr}")
                dynamatic_elapsed_time = (dynamatic_end - dynamatic_begin).total_seconds()
                logger.debug(f"Dynamatic elapsed time: {dynamatic_elapsed_time}")

            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("Synthesis process timed out and was killed")
                stdout, stderr = process.communicate()
                logger.debug(f"Partial output: {stdout}")
                logger.debug(f"Partial error: {stderr}")
                return 1, "Synthesis process timed out", dynamatic_elapsed_time, java_elapsed_time
            
            # Only proceed to the next step if synthesis was successful
            if process.returncode != 0:
                logger.error("Synthesis process failed")
                return 1, "Synthesis process failed", dynamatic_elapsed_time, java_elapsed_time

            # If rapidwright is False, return after dynamatic synthesis
            if not kernel_object._rapidwright:
                # Instead of chmod, use Docker to set permissions from inside the container
                fix_perm_cmd = f"docker exec dynarapid_container bash -c 'chmod -R a+rwx {container_path}'"
                try:
                    subprocess.run(fix_perm_cmd, shell=True, check=True, timeout=30)
                    logger.debug("Set permissions on output files via Docker")
                except Exception as e:
                    logger.error(f"Error setting permissions via Docker: {e}")
                
                # Return after dynamatic synthesis
                return 0, stdout, dynamatic_elapsed_time, java_elapsed_time

            # Only run the Java command if rapidwright is True
            logger.info("Running Java GenerateDesign command...")
            
            # Define the target directory for this attempt with new structure
            attempt_dir = f"/mount/results/solutions/dynarapid/{kernel_object._model}/{kernel_object._iter+1}/{kernel_object.kernel_name}/attempts_{attempt+1}/dcp_designs"
            
            # Create the target directory
            mkdir_cmd = f"docker exec dynarapid_container bash -c 'mkdir -p {attempt_dir}'"
            try:
                subprocess.run(mkdir_cmd, shell=True, check=True, timeout=30)
                logger.debug(f"Created directory for attempt {attempt}: {attempt_dir}")
            except Exception as e:
                logger.error(f"Error creating attempt directory: {e}")
                return 1, "Error creating attempt directory", dynamatic_elapsed_time, java_elapsed_time
            
            # Debug: Check if /DynaRapid exists and what it contains - with correct capitalization
            debug_cmd = f"docker exec dynarapid_container bash -c 'ls -la /DynaRapid/ || echo \"Directory does not exist\"'"
            try:
                debug_result = subprocess.run(debug_cmd, shell=True, check=False, capture_output=True, text=True)
                logger.debug(f"Debug /DynaRapid/ directory: {debug_result.stdout}")
            except Exception as e:
                logger.error(f"Error debugging DynaRapid directory: {e}")
            
            # Remove any existing symbolic link - with correct capitalization
            rm_link_cmd = f"docker exec dynarapid_container bash -c 'rm -rf /DynaRapid/designs'"
            try:
                subprocess.run(rm_link_cmd, shell=True, check=True, timeout=30)
                logger.debug("Removed existing symbolic link if any")
            except Exception as e:
                logger.error(f"Error removing existing link: {e}")
                # Continue anyway as it might not exist
            
            # Create a new symbolic link for this attempt - with correct capitalization
            link_cmd = f"docker exec dynarapid_container bash -c 'ln -sf {attempt_dir} /DynaRapid/designs'"
            try:
                subprocess.run(link_cmd, shell=True, check=True, timeout=30)
                logger.debug(f"Created symbolic link from /DynaRapid/designs to {attempt_dir}")
                
                # Verify the link was created properly - with correct capitalization
                verify_cmd = f"docker exec dynarapid_container bash -c 'ls -la /DynaRapid/designs'"
                verify_result = subprocess.run(verify_cmd, shell=True, check=False, capture_output=True, text=True)
                logger.debug(f"Symbolic link verification: {verify_result.stdout}")
            except Exception as e:
                logger.error(f"Error creating symbolic link: {e}")
                return 1, "Error creating symbolic link", dynamatic_elapsed_time, java_elapsed_time
            
            # Now run the Java command with the symbolic link in place
            java_cmd = f"docker exec dynarapid_container bash -c 'cd {container_path} && java ch.agsl.dynarapid.GenerateDesign -placer greedy -f {container_path}/reports/{kernel_object.kernel_name}.dot'"
            
            java_begin = datetime.now()
            java_process = subprocess.Popen(java_cmd, shell=True, stdout=subprocess.PIPE, 
                                           stderr=subprocess.PIPE, text=True)
            
            try:
                java_stdout, java_stderr = java_process.communicate(timeout=300)  # 5 minutes timeout
                java_end = datetime.now()
                logger.debug(f"Java command exit code: {java_process.returncode}")
                logger.debug(f"Java command output: {java_stdout}")
                logger.debug(f"Java command error: {java_stderr}")
                java_elapsed_time = (java_end - java_begin).total_seconds()
                logger.debug(f"Java command elapsed time: {java_elapsed_time}")
                
                # Append java output to previous output
                stdout += "\n\n--- Generate Design Output ---\n" + java_stdout
                
                # Check if files were actually created
                check_files_cmd = f"docker exec dynarapid_container bash -c 'find {attempt_dir} -type f | wc -l'"
                check_result = subprocess.run(check_files_cmd, shell=True, check=False, capture_output=True, text=True)
                file_count = check_result.stdout.strip()
                logger.debug(f"Files created in attempt directory: {file_count}")
                
                if file_count == "0":
                    # Try to find where files might have been created instead - with correct capitalization
                    find_files_cmd = f"docker exec dynarapid_container bash -c 'find /DynaRapid -type f -mmin -5 | grep -v \".class\"'"
                    find_result = subprocess.run(find_files_cmd, shell=True, check=False, capture_output=True, text=True)
                    logger.debug(f"Recent files in /DynaRapid: {find_result.stdout}")
                    
                    # If files were found elsewhere, try to copy them to our target directory
                    if find_result.stdout.strip():
                        logger.info("Found files in a different location, copying to target directory")
                        copy_cmd = f"docker exec dynarapid_container bash -c 'find /DynaRapid -type f -mmin -5 | grep -v \".class\" | xargs -I{{}} cp {{}} {attempt_dir}/'"
                        subprocess.run(copy_cmd, shell=True, check=False)
                
            except subprocess.TimeoutExpired:
                java_process.kill()
                logger.warning("Java process timed out and was killed")
                java_stdout, java_stderr = java_process.communicate()
                logger.debug(f"Partial Java output: {java_stdout}")
                logger.debug(f"Partial Java error: {java_stderr}")
                return 1, "Java GenerateDesign process timed out", dynamatic_elapsed_time, java_elapsed_time
            finally:
                # Always remove the symbolic link, regardless of success or failure - with correct capitalization
                rm_link_cmd = f"docker exec dynarapid_container bash -c 'rm -f /DynaRapid/designs'"
                try:
                    subprocess.run(rm_link_cmd, shell=True, check=True, timeout=30)
                    logger.debug("Removed symbolic link after Java command")
                except Exception as e:
                    logger.error(f"Error removing symbolic link: {e}")
                    # Continue anyway

            # Instead of chmod, use Docker to set permissions from inside the container
            fix_perm_cmd = f"docker exec dynarapid_container bash -c 'chmod -R a+rwx {container_path} {attempt_dir}'"
            try:
                subprocess.run(fix_perm_cmd, shell=True, check=True, timeout=30)
                logger.debug("Set permissions on output files via Docker")
            except Exception as e:
                logger.error(f"Error setting permissions via Docker: {e}")
                
        except subprocess.TimeoutExpired:
            logger.warning("A command timed out")
            return 1, "Command execution timed out", dynamatic_elapsed_time, java_elapsed_time
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return 1, "Unexpected error", dynamatic_elapsed_time, java_elapsed_time
        finally:
            # Make sure to clean up the symbolic link in case it wasn't done earlier - with correct capitalization
            rm_link_cmd = f"docker exec dynarapid_container bash -c 'rm -f /DynaRapid/designs'"
            try:
                subprocess.run(rm_link_cmd, shell=True, check=True, timeout=30)
                logger.debug("Final cleanup of symbolic link")
            except Exception as e:
                logger.error(f"Error in final cleanup of symbolic link: {e}")
                # Continue anyway
            
    finally:
        # Restore original umask
        os.umask(old_umask)
    
    return 0, stdout, dynamatic_elapsed_time, java_elapsed_time

def check_dot_dcp(kernel_object: hls.HLSEval, attempt: int=0, dynarapid_feedback = None):
    new_solution_path = f"{kernel_object.solution_path}/attempts_{attempt+1}"


    dot_path  = f"{new_solution_path}/dcp_designs/{kernel_object.kernel_name}.dot"
    dot_path2 = f"{new_solution_path}/dcp_designs/{kernel_object.kernel_name}/{kernel_object.kernel_name}.dot"
    dcp_path  = f"{new_solution_path}/dcp_designs/{kernel_object.kernel_name}_routed.dcp"
    dcp_path2 = f"{new_solution_path}/dcp_designs/{kernel_object.kernel_name}/{kernel_object.kernel_name}_routed.dcp"


    check_path_dot1 = pathlib.Path(dot_path)
    check_path_dot1 = check_path_dot1.resolve()
    check_path_dot2 = pathlib.Path(dot_path2)
    check_path_dot2 = check_path_dot2.resolve()
    check_path_dcp1 = pathlib.Path(dcp_path)
    check_path_dcp1 = check_path_dcp1.resolve()
    check_path_dcp2 = pathlib.Path(dcp_path2)
    check_path_dcp2 = check_path_dcp2.resolve()

    dynarapid_feedback = {}
    dynarapid_feedback['compile_feedback'] = {}
    logger.info(dot_path)
    logger.info(check_path_dot1.exists())
    logger.info(dot_path2)
    logger.info(check_path_dot2.exists())
    logger.info(dcp_path)
    logger.info(check_path_dcp1.exists())
    logger.info(dcp_path2)
    logger.info(check_path_dcp2.exists())


    dynarapid_feedback['compile_feedback']["has_dot"] = True if check_path_dot1.exists() or check_path_dot2.exists() else False
    dynarapid_feedback['compile_feedback']["has_dcp"] = True if check_path_dcp1.exists() or check_path_dcp2.exists() else False


    out_data_path = f"{new_solution_path}/{kernel_object.kernel_name}_data.json"

    parser.save_feedback_data(out_data_path, dynarapid_feedback)
    return dynarapid_feedback
