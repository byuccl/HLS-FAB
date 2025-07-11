import json
import os

kernels_dir = "kernels/"
output_file = "hlseval.jsonl"
# kernel_testbench_includes = "kernel_testbench_includes.h"
kernel_name_list = "kernel_list.txt"
# test_map_header = "test_function_map.h"

kernel_names = []

# Open all files for writing (overwrite)
with open(output_file, "w") as outfile, open(kernel_name_list, "w") as namefile:

    for kernel_folder in os.listdir(kernels_dir):
        kernel_json = os.path.join(kernels_dir, kernel_folder, "kernel.json")
        if os.path.exists(kernel_json):
            with open(kernel_json, "r") as infile:
                data = json.load(infile)
                json_string = json.dumps(data, separators=(",", ":"))
                outfile.write(json_string + "\n")

                kernel_name = data.get("kernel_name")
                if kernel_name:
                    kernel_names.append(kernel_name)
                    # kernel_includes.write(
                    #     f'#include "kernels/{kernel_name}/testbench.h"\n'
                    # )
                    namefile.write(kernel_name + "\n")
