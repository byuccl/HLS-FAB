import os
import subprocess
import logging
import time
import json
from datetime import datetime
import hlseval

logger = logging.getLogger("DYNARAPID")

def run_dynarapid(kernel_name, solutions_path, attempt_num, model, rapidwright=False):
    dynamatic_elapsed_time = None
    java_elapsed_time = None
    # Extract the relevant part of the path
    # Get the path from solutions_path that matches the container's structure
    if "/results/solutions/dynarapid/" in solutions_path:
        # Extract the path starting from /results/...
        container_path = "/mount" + solutions_path[solutions_path.find("/results/solutions/dynarapid/"):]
    else:
        # Use the full path as fallback
        container_path = f"/mount{solutions_path}"
    
    logger.debug(f"Host path: {solutions_path}")
    logger.debug(f"Container path: {container_path}")
    
    # Create the TCL script content
    syn_tcl = f"""
set_project .
set_top_file {kernel_name}.cpp
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
        filename = f"{solutions_path}/synthesis.tcl"
        try:
            with open(filename, "w") as file:
                file.write(syn_tcl)
            logger.debug(f"Successfully wrote to {filename}")
        except Exception as e:
            logger.error(f"An error occurred while writing TCL script: {e}")
            return 1, "No feedback at this time", dynamatic_elapsed_time, java_elapsed_time
        
        # Prepare the source directory with proper permissions
        try:
            os.makedirs(f"{solutions_path}/src", mode=0o777, exist_ok=True)
            os.makedirs(f"{solutions_path}/reports", mode=0o777, exist_ok=True)
            
            # Use the cp command with the --preserve=mode option to preserve mode but not ownership
            subprocess.run(f"cp {solutions_path}/{kernel_name}.cpp {solutions_path}/src/", 
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
            if not rapidwright:
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
            attempt_dir = f"/mount/results/dcp_designs/{model}/{kernel_name}/attempts_{attempt_num+1}"
            
            # Create the target directory
            mkdir_cmd = f"docker exec dynarapid_container bash -c 'mkdir -p {attempt_dir}'"
            try:
                subprocess.run(mkdir_cmd, shell=True, check=True, timeout=30)
                logger.debug(f"Created directory for attempt {attempt_num}: {attempt_dir}")
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
            rm_link_cmd = f"docker exec dynarapid_container bash -c 'rm -f /DynaRapid/designs'"
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
            java_cmd = f"docker exec dynarapid_container bash -c 'cd {container_path} && java ch.agsl.dynarapid.GenerateDesign -placer greedy -f {container_path}/reports/{kernel_name}.dot'"
            
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
