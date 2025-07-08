import os
import subprocess
import logging
import time
import json
from datetime import datetime
import hlseval

logger = logging.getLogger("DYNARAPID")

def run_dynarapid(kernel_name, solutions_path):
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
            return 1, "No feedback at this time", 0
        
        # Prepare the source directory with proper permissions
        try:
            os.makedirs(f"{solutions_path}/src", mode=0o777, exist_ok=True)
            os.makedirs(f"{solutions_path}/reports", mode=0o777, exist_ok=True)
            
            # Use the cp command with the --preserve=mode option to preserve mode but not ownership
            subprocess.run(f"cp {solutions_path}/{kernel_name}.cpp {solutions_path}/src/", 
                          shell=True, check=True, text=True)
            
        except Exception as e:
            logger.error(f"Error setting up directories: {e}")
            return 1, "No feedback at this time", 0

        # Start the container if it's not already running
        try:
            # Capture and discard the output from docker start
            result = subprocess.run("docker start dynarapid_container", 
                            shell=True, check=True, text=True, 
                            capture_output=True)  # Add capture_output=True
            logger.info("Started dynarapid_container")
            
            # Verify the container is running
            container_check = subprocess.run("docker ps | grep dynarapid_container", 
                                           shell=True, text=True, capture_output=True)
            logger.debug(f"Container status: {container_check.stdout}")
            
            if not container_check.stdout.strip():
                logger.error("Error: Container not running properly")
                return 1, "Container not running", 0
                
        except Exception as e:
            logger.error(f"Error with container operations: {e}")
            return 1, "Container operation failed", 0
        
        # Execute commands separately with timeout protection
        try:
            # Test if we can execute basic commands
            test_cmd = f"docker exec dynarapid_container bash -c 'ls -la {container_path}'"
            test_result = subprocess.run(test_cmd, shell=True, text=True, 
                                      capture_output=True, timeout=10)
            logger.debug(f"Test command output: {test_result.stdout}")
            logger.debug(f"Test command error: {test_result.stderr}")
            
            if test_result.returncode != 0:
                logger.error("Error: Cannot access the directory in the container")
                return 1, "Directory access failed", 0
            
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
                dynamatic_elapsed_time = dynamatic_end-dynamatic_begin
                logger.debug(f"Dynamatid elapsed time: {dynamatic_elapsed_time}")

            except subprocess.TimeoutExpired:
                process.kill()
                logger.warning("Synthesis process timed out and was killed")
                stdout, stderr = process.communicate()
                logger.debug(f"Partial output: {stdout}")
                logger.debug(f"Partial error: {stderr}")

            # Instead of chmod, use Docker to set permissions from inside the container
            # This works because Docker is likely running with sufficient privileges
            fix_perm_cmd = f"docker exec dynarapid_container bash -c 'chmod -R a+rwx {container_path}'"
            try:
                subprocess.run(fix_perm_cmd, shell=True, check=True, timeout=30)
                logger.debug("Set permissions on output files via Docker")
            except Exception as e:
                logger.error(f"Error setting permissions via Docker: {e}")
                
        except subprocess.TimeoutExpired:
            logger.warning("A command timed out")
            return 1, "Command execution timed out", 0
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}")
            return 1, "Unexpected error", 0
            
    finally:
        # Restore original umask
        os.umask(old_umask)
    
    return 0, stdout, dynamatic_elapsed_time
