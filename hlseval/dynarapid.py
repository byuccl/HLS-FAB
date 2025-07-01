import os
import subprocess
import logging
import time

logger = logging.getLogger("__main__")

def run_dynarapid(kernel_name, solutions_path):
    # Extract the relevant part of the path
    # Get the path from solutions_path that matches the container's structure
    if "/results/solutions/dynarapid/" in solutions_path:
        # Extract the path starting from /results/...
        container_path = "/mount" + solutions_path[solutions_path.find("/results/solutions/dynarapid/"):]
    else:
        # Use the full path as fallback
        container_path = f"/mount{solutions_path}"
    
    print(f"Host path: {solutions_path}")
    print(f"Container path: {container_path}")
    
    # Create the TCL script content
    syn_tcl = f"""
set_project .
set_top_file {kernel_name}.cpp
synthesize -use-lsq=false -simple-buffers=true -verbose
#cdfg
write_hdl

exit
    """
    
    # Write the TCL script to a file
    filename = f"{solutions_path}/synthesis.tcl"
    try:
        with open(filename, "w") as file:
            file.write(syn_tcl)
        print(f"Successfully wrote to {filename}")
    except Exception as e:
        print(f"An error occurred while writing TCL script: {e}")
        return 1, "No feedback at this time"
    
    # Prepare the source directory
    try:
        os.makedirs(f"{solutions_path}/src", exist_ok=True)
        subprocess.run(f"cp {solutions_path}/{kernel_name}.cpp {solutions_path}/src/", 
                      shell=True, check=True, text=True)
        os.makedirs(f"{solutions_path}/reports", exist_ok=True)
    except Exception as e:
        print(f"Error setting up directories: {e}")
        return 1, "No feedback at this time"
    
    # Start the container if it's not already running
    try:
        subprocess.run("docker start dynarapid_container", 
                      shell=True, check=True, text=True)
        print("Started dynarapid_container")
        
        # Verify the container is running
        container_check = subprocess.run("docker ps | grep dynarapid_container", 
                                       shell=True, text=True, capture_output=True)
        print(f"Container status: {container_check.stdout}")
        
        if not container_check.stdout.strip():
            print("Error: Container not running properly")
            return 1, "Container not running"
            
    except Exception as e:
        print(f"Error with container operations: {e}")
        return 1, "Container operation failed"
    
    # Execute commands separately with timeout protection
    try:
        # Test if we can execute basic commands
        test_cmd = f"docker exec dynarapid_container bash -c 'ls -la {container_path}'"
        test_result = subprocess.run(test_cmd, shell=True, text=True, 
                                  capture_output=True, timeout=10)
        print(f"Test command output: {test_result.stdout}")
        print(f"Test command error: {test_result.stderr}")
        
        if test_result.returncode != 0:
            print("Error: Cannot access the directory in the container")
            return 1, "Directory access failed"
        
        # Run the dynamatic command
        print("Running dynamatic synthesis...")
        cmd = f"docker exec dynarapid_container bash -c 'cd {container_path} && dynamatic synthesis.tcl'"
        
        # Use Popen to have more control over the process
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, 
                                stderr=subprocess.PIPE, text=True)
        
        # Wait with timeout
        try:
            stdout, stderr = process.communicate(timeout=300)  # 5 minutes timeout
            print(f"Synthesis exit code: {process.returncode}")
            print(f"Synthesis output: {stdout}")
            print(f"Synthesis error: {stderr}")
        except subprocess.TimeoutExpired:
            process.kill()
            print("Synthesis process timed out and was killed")
            stdout, stderr = process.communicate()
            print(f"Partial output: {stdout}")
            print(f"Partial error: {stderr}")
        
        # # Check for .dot files in various locations
        # print("Searching for dot files...")
        # locations = [
        #     container_path,
        #     os.path.dirname(container_path),
        #     f"{container_path}/.."
        # ]
        
        # dot_files_found = False
        # for location in locations:
        #     find_cmd = f"docker exec dynarapid_container bash -c 'find {location} -name \"*{kernel_name}*.dot\" -print 2>/dev/null || echo \"No files found\"'"
        #     dot_files = subprocess.run(find_cmd, shell=True, text=True, 
        #                             capture_output=True, timeout=10)
            
        #     if dot_files.stdout.strip() and "No files found" not in dot_files.stdout:
        #         dot_files_found = True
        #         print(f"Found .dot files in {location}: {dot_files.stdout}")
                
        #         # Copy to reports directory
        #         for dot_file in dot_files.stdout.strip().split('\n'):
        #             if dot_file and "No files found" not in dot_file:
        #                 base_name = os.path.basename(dot_file)
        #                 move_cmd = f"docker exec dynarapid_container bash -c 'cp {dot_file} {container_path}/reports/{base_name}'"
        #                 cp_result = subprocess.run(move_cmd, shell=True, text=True, 
        #                                        capture_output=True, timeout=10)
        #                 print(f"Copy result: {cp_result.stdout} {cp_result.stderr}")
        
        # if not dot_files_found:
        #     print("No .dot files found in any location!")
        #     return 1, "No dot files generated"
        
        # # Run write_hdl command
        # print("Running write_hdl...")
        # write_cmd = f"docker exec dynarapid_container bash -c 'cd {container_path}/reports && dynamatic write_hdl'"
        # write_result = subprocess.run(write_cmd, shell=True, text=True, 
        #                            capture_output=True, timeout=60)
        # print(f"Write HDL exit code: {write_result.returncode}")
        # print(f"Write HDL output: {write_result.stdout}")
        # print(f"Write HDL error: {write_result.stderr}")
        
        # # Check for generated HDL files
        # find_hdl_cmd = f"docker exec dynarapid_container bash -c 'find {container_path} -name \"*.v*\" -o -name \"*.vhd*\" -print'"
        # hdl_files = subprocess.run(find_hdl_cmd, shell=True, text=True, 
        #                         capture_output=True, timeout=10)
        # print(f"Found HDL files: {hdl_files.stdout}")
        
        # if not hdl_files.stdout.strip():
        #     return 1, "No HDL files generated"
            
    except subprocess.TimeoutExpired:
        print("A command timed out")
        return 1, "Command execution timed out"
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return 1, "Unexpected error"
    
    return 0, "Successfully ran DynaRapid"