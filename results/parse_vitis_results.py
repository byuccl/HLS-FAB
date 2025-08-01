import os
import json
import csv
import re
from pathlib import Path

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
        llm_time = row[1].strip() if len(row) > 2 else '-'
        csim_time = row[3].strip() if len(row) > 3 else '-'
        csynth_time = row[4].strip() if len(row) > 4 else '-'
        
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

def create_csv(base_dir, output_file):
    """Process all data and create a CSV file."""
    # Define the models to process
    with open("./dataset/llm_models.txt", "r") as f:
        lines = f.readlines()
    models = [line.strip() for line in lines]
    
    # Define CSV headers
    headers = [
        'model', 'iteration', 'kernel', 'attempts', 'llm_time', 'status', 'match', 'errors', 'csim_time',
        'synth_errors', 'fmax', 'latency', 'bram', 'dsp', 'ff', 'lut', 'csynth_time'
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
        
        # Process each model
        for model in models:
            model_dir = base_path / model
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
            for iteration in iterations:
                iteration_dir = model_dir / iteration
                
                # Iteration might have its own timing log
                iteration_timing_log_path = iteration_dir / 'logs' / 'timing.log'
                iteration_timing_data = parse_timing_log(str(iteration_timing_log_path))
                
                # Combine timing data, with iteration data taking precedence
                timing_data = {**model_timing_data, **iteration_timing_data}
                
                # Get all kernel directories for this iteration
                kernel_dirs = get_all_kernel_dirs(iteration_dir)
                
                # Process each kernel
                for kernel_dir in kernel_dirs:
                    kernel_path = iteration_dir / kernel_dir
                    
                    if not kernel_path.exists():
                        print(f"Skipping kernel directory (not found): {kernel_path}")
                        continue
                    
                    print(f"  Processing iteration: {iteration}, kernel: {kernel_dir}")
                    
                    # Process each attempt
                    rows_for_kernel = []
                    for attempt in range(1, 2):
                        attempt_dir = kernel_path / f"attempts_{attempt}"
                        if not attempt_dir.exists():
                            print(f"    Skipping attempt directory (not found): {attempt_dir}")
                            continue
                        
                        print(f"    Processing attempt: {attempt}")
                        
                        data_file = attempt_dir / f"{kernel_dir}_data.json"

                        # Parse JSON data
                        try:
                            with open(data_file, 'r') as f:
                                data = json.load(f)
                        except FileNotFoundError:
                            print(f"      Skipping data file (not found): {data_file}")
                            data = {}
                        except json.JSONDecodeError as e:
                            print(f"      ERROR: Failed to parse JSON in {data_file}: {e}")
                            continue
                        
                        # Extract data
                        status = data.get('sim_feedback', {}).get('status', '')
                        match = data.get('sim_feedback', {}).get('match', '')
                        
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
                        llm_time = timing_data.get(timing_key, {}).get('llm_time', '-')
                        csim_time = timing_data.get(timing_key, {}).get('csim_time', '-')
                        csynth_time = timing_data.get(timing_key, {}).get('csynth_time', '-')
                        
                        # Extract synthesis data
                        fmax = data.get('synth_feedback', {}).get('fmax', '-')
                        
                        latency_data = data.get('synth_feedback', {}).get('latency', {})
                        if latency_data and isinstance(latency_data, dict):
                            if 'max' in latency_data and latency_data['max'] is not None:
                                latency = latency_data['max']
                            else:
                                latency = '-'
                        else:
                            latency = 'null'
                        
                        resources = data.get('synth_feedback', {}).get('resources', {})
                        bram = resources.get('bram', '-')
                        dsp = resources.get('dsp', '-')
                        ff = resources.get('ff', '-')
                        lut = resources.get('lut', '-')
                        
                        # Create row for CSV
                        row = [
                            model,
                            iteration,
                            kernel_dir,
                            attempt,
                            llm_time,
                            status,
                            match,
                            errors,
                            csim_time,
                            synth_errors_str,
                            fmax,
                            latency,
                            bram,
                            dsp,
                            ff,
                            lut,
                            csynth_time
                        ]
                        
                        rows_for_kernel.append(row)
                    
                    # Write all rows for this kernel
                    for row in rows_for_kernel:
                        writer.writerow(row)
    
    print(f"CSV file created successfully: {output_file}")

# Execute the script
base_directory = "./results/solutions/vitis"
output_csv = "results_summary.csv"
create_csv(base_directory, output_csv)
