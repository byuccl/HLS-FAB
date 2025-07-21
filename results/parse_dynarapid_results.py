import os
import csv
from pathlib import Path

def search_files(base_dir):
    """
    Search for .dot and _routed.dcp files in the given directory structure
    and return information about their existence.
    """
    models = ['gpt-4.1', 'gpt-4o']
    kernels = [
        'conv2d_int32',
        'argmax_int32',
        'matmul_16x16_int32',
        'pad2d_int32',
        'bitcount_uint32',
        'negate_int32',
        'sign_int32',
        'hardsigmoid_int32',
        'bitwisexor_uint32',
        'compare_gt_int32',
        'vectormatrix_mult_int32',
        'compare_lt_int32',
        'dotproduct_bias_relu_int32'
    ]
    
    results = []
    
    for model in models:
        model_path = Path(base_dir) / 'results' / 'dcp_designs' / model
        if not model_path.exists():
            print(f"Warning: Path does not exist: {model_path}")
            continue
            
        for kernel in kernels:
            kernel_path = model_path / kernel
            if not kernel_path.exists():
                print(f"Warning: Path does not exist: {kernel_path}")
                continue
                
            for attempt in range(4):  # 0 to 3
                attempt_path = kernel_path / f"attempt_{attempt}"
                if not attempt_path.exists():
                    print(f"Warning: Path does not exist: {attempt_path}")
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
                    'has_dcp': has_dcp
                })
    
    return results

def write_csv(results, output_file):
    """Write the results to a CSV file."""
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['model', 'kernel', 'attempt', 'has_dot', 'has_dcp']
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
