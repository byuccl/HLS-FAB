# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

import os
import json
import csv
import re
from pathlib import Path
import argparse


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
        csim_time = row[3].strip() if len(row) > 3 else None
        csynth_time = row[4].strip() if len(row) > 4 else None
        
        timing_data[key] = {
            'llm_time'   : llm_time,
            'csim_time'  : csim_time,
            'csynth_time': csynth_time
        }
    return timing_data

def get_iterations(model_dir):
    """Get all iteration directories within a model directory."""
    if not model_dir.exists():
        return []
    
    iterations = []
    for item in model_dir.iterdir():
        if item.is_dir() and item.name != "logs":
            iterations.append(item.name)
    
    return iterations

def get_all_kernel_dirs(iteration_dir):
    """Get all kernel directories within an iteration directory."""
    if not iteration_dir.exists():
        return []
    
    kernel_dirs = []
    for item in iteration_dir.iterdir():
        if item.is_dir() and "attempts_" not in item.name and "logs" != item.name:
            kernel_dirs.append(item.name)
    
    return kernel_dirs

def create_csv(base_dir, output_file, runs, attempts):
    """Process all data and create a CSV file."""
    # Define the models to process same as the most recent completions run
    with open("./dataset/llm_models.txt", "r") as f:
        lines = f.readlines()
    models = [line.strip() for line in lines]

    # Define the kernels to process same as the dataset/kernels directory
    with open("./dataset/vitis_general_kernel_list.txt", "r") as f:
        kernels = f.readlines()
    kernels = [kernel.strip() for kernel in kernels]
    
    # Define CSV headers
    headers = [
        'model', 'RAG', 'kernel', 'run', 'attempt', 'compilation_status', 'outputs_match', 'csim_success',
        'fmax', 'latency', 'bram', 'dsp', 'ff', 'lut', 'csynth_success',
        'llm_time', 'csim_time', 'csynth_time', 'csim_errors', 'synth_errors'
    ]
    
    base_path = Path(base_dir)
    
    # Debug info
    print(f"Base directory: {base_path}")
    if not base_path.exists():
        print(f"ERROR: Base directory does not exist: {base_path}")
        return
    
    # Open CSV file for writing
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        for rag in ("rag", "no_rag"):
            rag_path = base_path / rag
            # Process each model
            for model in models:
                model_dir = rag_path / model
                if not model_dir.exists():
                    print(f"Skipping model directory (not found): {model_dir}")
                    continue
                
                print(f"Processing model: {model}")
                
                # Parse timing log (now might be in different locations depending on structure)
                model_timing_log_path = model_dir / 'logs' / 'timing.log'
                model_timing_data = parse_timing_log(str(model_timing_log_path))
                
                # Get all iteration directories for this model
                iterations = get_iterations(model_dir)
                
                # Process each iteration
                for iteration in range(1, runs+1):
                    iteration_dir = model_dir / str(iteration)
                    print(iteration_dir)
                    # Iteration might have its own timing log
                    iteration_timing_log_path = iteration_dir / 'logs' / 'timing.log'
                    iteration_timing_data = parse_timing_log(str(iteration_timing_log_path))
                    
                    # Combine timing data, with iteration data taking precedence
                    timing_data = {**model_timing_data, **iteration_timing_data}
                    
                    # Get all kernel directories for this iteration
                    kernel_dirs = kernels
                    
                    # Process each kernel
                    for kernel_dir in kernel_dirs:
                        kernel_path = iteration_dir / kernel_dir
                        
                        if not kernel_path.exists():
                            # print(f"Skipping kernel directory (not found): {kernel_path}")
                            continue
                        
                        # print(f"  Processing iteration: {iteration}, kernel: {kernel_dir}")
                        
                        # Process each attempt
                        rows_for_kernel = []
                        for attempt in range(1, attempts+1):
                            attempt_dir = kernel_path / f"attempts_{attempt}"
                            if not attempt_dir.exists():
                                print(f"    Skipping attempt directory (not found): {attempt_dir}")
                                continue
                            
                            # print(f"    Processing attempt: {attempt}")
                            
                            data_file = attempt_dir / f"{kernel_dir}_data.json"

                            # Parse JSON data
                            try:
                                with open(data_file, 'r') as f:
                                    data = json.load(f)
                            except FileNotFoundError:
                                # print(f"      Skipping data file (not found): {data_file}")
                                data = {}
                            except json.JSONDecodeError as e:
                                print(f"      ERROR: Failed to parse JSON in {data_file}: {e}")
                                continue
                            
                            # Extract data
                            status = data.get('sim_feedback', {}).get('status', '')
                            match = data.get('sim_feedback', {}).get('match', '')

                            csim_success = (match != "mismatch")
                            
                            # Extract errors as string
                            csim_errors = data.get('sim_feedback', {}).get('errors', [])
                            if csim_errors and isinstance(csim_errors, list) and len(csim_errors) > 0:
                                errors = f'"{csim_errors[0]}"'
                            else:
                                errors = '-'
                            
                            # Extract synthesis errors
                            synth_errors = data.get('synth_feedback', {}).get('errors', [])
                            if synth_errors and isinstance(synth_errors, list) and len(synth_errors) > 0:
                                synth_errors_str = f'"{synth_errors[0]}"'
                            else:
                                synth_errors_str = '-'
                            
                            # Extract timing data
                            timing_key = f"{kernel_dir}_{attempt}"
                            llm_time = timing_data.get(timing_key, {}).get('llm_time', None)
                            csim_time = timing_data.get(timing_key, {}).get('csim_time', None)
                            csynth_time = timing_data.get(timing_key, {}).get('csynth_time', None)
                            
                            # Extract synthesis data
                            fmax = data.get('synth_feedback', {}).get('fmax', None)
                            
                            latency_data = data.get('synth_feedback', {}).get('latency', {})
                            if latency_data and isinstance(latency_data, dict):
                                if 'max' in latency_data and latency_data['max'] is not None:
                                    latency = latency_data['max']
                                else:
                                    latency = None
                            else:
                                latency = None
                            
                            resources = data.get('synth_feedback', {}).get('resources', {})
                            bram = resources.get('bram', None)
                            dsp = resources.get('dsp', None)
                            ff = resources.get('ff', None)
                            lut = resources.get('lut', None)

                            csynth_success = (latency is not None and latency > 5)

                            
                            # Create row for CSV
                            row = [
                                model,
                                rag,
                                kernel_dir,
                                iteration,
                                attempt,
                                status,
                                match,
                                csim_success,
                                fmax,
                                latency,
                                bram,
                                dsp,
                                ff,
                                lut,
                                csynth_success,
                                llm_time,
                                csim_time,
                                csynth_time,
                                errors,
                                synth_errors_str
                            ]
                            
                            rows_for_kernel.append(row)
                        
                        # Write all rows for this kernel
                        for row in rows_for_kernel:
                            writer.writerow(row)
    
    print(f"CSV file created successfully: {output_file}")

# Execute the script
base_directory = "./results/"
# base_directory = "./results/solutions/vitis/no_rag" #TODO: Make this operate dynamically
output_csv = "vitis_results.csv"
if __name__ == "__main__":

    # Arguments set up
    parser = argparse.ArgumentParser(description="Run HLS kernel processing.")
    parser.add_argument("--target", dest='target', type=str, help="Target folder to search and parse data", default="solutions")
    parser.add_argument("--runs", dest='runs', type=int, help="Number of runs that were done (k or iterations)", default=1)
    parser.add_argument("--attempts",  dest='attempts', type=int, help="Number of feedback attempts (n)", default=1)

    args = parser.parse_args()
    runs = args.runs
    attempts = args.attempts

    target = args.target
    base_directory = base_directory + target + "/vitis"
    create_csv(base_directory, output_csv, runs, attempts)
