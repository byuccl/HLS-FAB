import os
import csv
from pathlib import Path

def search_files(base_dir):
    """
    Search for .dot and _routed.dcp files in the given directory structure
    and return information about their existence.
    """
    with open("./dataset/llm_models.txt", "r") as f:
        lines = f.readlines()
    models = [line.strip() for line in lines]
    with open("./dataset/kernel_list.txt", "r") as f:
        kernels = f.readlines()
    kernels = [kernel.strip() for kernel in kernels]
    
    results = []
    
    for model in models:
        base_path = Path(base_dir)
        model_path = base_path / 'results/solutions/dynarapid' / model 
        if not model_path.exists():
            print(f"Warning 1: Path does not exist: {model_path}")
            continue
        for p in range(3):
            p_path = model_path / f"{p+1}"
            if not model_path.exists():
                print(f"Warning 2: Path does not exist: {model_path}")
                continue
            for kernel in kernels:
                kernel_path = p_path / kernel
                if not kernel_path.exists():
                    print(f"Warning 3: Path does not exist: {kernel_path}")
                    continue
                    
                for attempt in range(3):  # 0 to 3
                    attempt_path = kernel_path / f"attempts_{attempt+1}" / "dcp_designs"
                    print(attempt_path)
                    if not attempt_path.exists():
                        print(f"Warning 4: Path does not exist: {attempt_path}")
                        continue
                    
                    # Check for kernel subdirectory
                    kernel_subdir = attempt_path / kernel
                    if kernel_subdir.exists():
                        search_path = kernel_subdir
                    else:
                        search_path = attempt_path
                    
                    # Look for the .dot file
                    dot_file = search_path / f"{kernel}.dot"
                    has_dot = dot_file.exists()
                    
                    # Look for the .dcp file
                    dcp_file = search_path / f"{kernel}_routed.dcp"
                    has_dcp = dcp_file.exists()
                    
                    # Record the result
                    results.append({
                        'model': model,
                        'kernel': kernel,
                        'attempt': attempt,
                        'has_dot': has_dot,
                        'has_dcp': has_dcp,
                        'pass@k': p+1
                    })
    
    return results

def write_csv(results, output_file):
    """Write the results to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'kernel', 'attempt', 'has_dot', 'has_dcp', 'pass@k']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
            
    print(f"Results written to {output_file}")

def main():
    base_dir = "."  # Change this to your base directory path if needed
    output_file = "design_files_status.csv"
    
    print("Searching for .dot and _routed.dcp files...")
    results = search_files(base_dir)
    
    print(f"Found {len(results)} entries")
    write_csv(results, output_file)

if __name__ == "__main__":
    main()
