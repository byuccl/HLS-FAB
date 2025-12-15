import argparse
import os
import csv
from pathlib import Path
import re

def parse_timing_log(log_path):
    """Parse the timing.log file to extract csim and csynth times."""
    timing_data = {}
    
    if not os.path.exists(log_path):
        print(f"Warning: Timing log not found at {log_path}")
        return timing_data
    
    with open(log_path, 'r') as f:
        content = f.read()
    
    # Extract rows from the timing table
    rows = re.findall(r'\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|\s+([^|]+)\s+\|', content)
    
    # Skip the header row
    for row in rows[1:]:
        kernel = row[0].strip()
        
        # Extract attempt number from kernel name if present
        kernel_parts = kernel.split('_')
        if kernel_parts[-2] == "attempt" and kernel_parts[-1].isdigit():
            attempt = int(kernel_parts[-1])
            kernel_name = "_".join(kernel_parts[:-2])
            key = f"{kernel_name}_{attempt}"
        else:
            # Use the kernel name directly if no attempt is in the name
            key = kernel
        llm_time = row[1].strip() if len(row) > 2 else None
        dynarapid_time = row[2].strip() if len(row) >3 else None
        csim_time = row[3].strip() if len(row) > 3 else None
        csynth_time = row[4].strip() if len(row) > 4 else None
        
        timing_data[key] = {
            'llm_time'   : llm_time,
            'csim_time'  : csim_time,
            'dynarapid_time': dynarapid_time,
            'csynth_time': csynth_time
        }
    return timing_data

def search_files(base_dir, k=1, attempts=1):
    """
    Search for .dot and _routed.dcp files in the given directory structure
    and return information about their existence.
    """
    with open("./dataset/llm_models.txt", "r") as f:
        lines = f.readlines()
    models = [line.strip() for line in lines]
    with open("./dataset/dynarapid_kernel_list.txt", "r") as f:
        kernels = f.readlines()
    kernels = [kernel.strip() for kernel in kernels]
    
    results = []
    for rag in ("rag", "no_rag"):
        base_path = Path(base_dir)
        rag_path = base_path / rag
        for model in models:
            model_path = rag_path / model 
            if not model_path.exists():
                print(f"Warning 1: Path does not exist: {model_path}")
                continue
            for p in range(k):
                p_path = model_path / f"{p+1}"
                if not model_path.exists():
                    print(f"Warning 2: Path does not exist: {model_path}")
                    continue
                iteration_timing_log_path = p_path / 'logs' / 'timing.log'
                iteration_timing_data = parse_timing_log(str(iteration_timing_log_path))
                    
                # Combine timing data, with iteration data taking precedence
                timing_data = iteration_timing_data
                # print(timing_data)
                for kernel in kernels:
                    kernel_path = p_path / kernel
                    if not kernel_path.exists():
                        print(f"Warning 3: Path does not exist: {kernel_path}")
                        continue
                        
                    for attempt in range(attempts):  # 0 to 3
                        attempt_path = kernel_path / f"attempts_{attempt+1}" / "dcp_designs"
                        # print(attempt_path)
                        if not attempt_path.exists():
                            print(f"Warning 4: Path does not exist: {attempt_path}")
                            continue
                        
                        # Check for kernel subdirectory
                        kernel_subdir = attempt_path / kernel
                        if kernel_subdir.exists():
                            search_path = kernel_subdir
                        else:
                            search_path = attempt_path
                        
                        timing_key = f"{kernel}_{attempt+1}"
                        
                        llm_time = timing_data.get(timing_key, {}).get('llm_time', None)

                        csim_time = timing_data.get(timing_key, {}).get('csim_time', None)
                        dynarapid_time = timing_data.get(timing_key, {}).get('dynarapid_time', None)          

                        # Look for the .dot file
                        dot_file = search_path / f"{kernel}.dot"
                        has_dot = dot_file.exists()
                        
                        # Look for the .dcp file
                        dcp_file = search_path / f"{kernel}_routed.dcp"
                        has_dcp = dcp_file.exists()
                        
                        # Record the result
                        results.append({
                            'model': model,
                            'rag': True if rag == "rag" else False,
                            'kernel': kernel,
                            'run': p+1,
                            'attempt': attempt+1,
                            'has_dot': has_dot,
                            'has_dcp': has_dcp,
                            'llm_time': llm_time,
                            'sim_time': csim_time,
                            'dynarapid_time': dynarapid_time
                        })
        
    return results

def write_csv(results, output_file):
    """Write the results to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'rag', 'kernel', 'run', 'attempt', 'has_dot', 'has_dcp', 'llm_time', 'sim_time', 'dynarapid_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
            
    print(f"Results written to {output_file}")

def main():


        # Arguments set up
    parser = argparse.ArgumentParser(description="Run HLS kernel processing.")
    parser.add_argument("--target", dest='target', type=str, help="Target folder to search and parse data", default="solutions")
    parser.add_argument("--runs", dest='runs', type=int, help="Number of runs that were done (k or iterations)", default=1)
    parser.add_argument("--attempts",  dest='attempts', type=int, help="Number of feedback attempts (n)", default=1)

    args = parser.parse_args()
    runs = args.runs
    attempts = args.attempts

    target = args.target
    base_dir = "."  # Change this to your base directory path if needed
    base_directory = "./results/" + target + "/dynarapid"


    
    output_file = "design_files_status.csv"
    
    print("Searching for .dot and _routed.dcp files...")
    results = search_files(base_directory, runs, attempts)
    
    print(f"Found {len(results)} entries")
    write_csv(results, output_file)

if __name__ == "__main__":
    main()
