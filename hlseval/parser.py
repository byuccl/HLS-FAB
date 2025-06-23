import re
import json
from datetime import datetime
import pathlib
import logging

logger = logging.getLogger("PARSER")

file_path_begin = "./results/solutions/"
in_file_path_mid = "/hls/syn/report/"
rpt_file_end = "_csynth.rpt"
log_folder_path = "/synth_logs/"
saved_log_file = "/logs/hls_compile.log"
saved_sim_log_file = "/logs/hls_run_csim.log"
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


models = {
    "gpt-4o",
}


# #open the kernels list to get the names of the kernels
# with open("./dataset/hlseval.jsonl", "r") as f:
#     tests = [json.loads(line) for line in f]
    
# Function to collect all the synthesis data and warnings/errors to be used in the feedback
def gather_feedback(synthesis_output=None, data=None, goals=None):
    """Function to gather feedback from synthesis output. Holds Errors and Warnings and Performance data."""
    if synthesis_output is None:
        synthesis_output = ""
    errors = re.findall(r"ERROR: \[.*?\] (.*)", synthesis_output)
    warnings = re.findall(r"WARNING: \[.*?\] (.*)", synthesis_output)
    
    if not data:
        data = {
            "Fmax": None,
            "lat_min": None,
            "lat_max": None,
            "inter_max": None,
            "bram": None,
            "dsp": None,
            "ff": None,
            "lut": None
        }
    
    feedback = {
        "errors": errors,
        "warnings": warnings,
        "fmax": data["Fmax"],
        "latency": {
            "min": data["lat_min"],
            "max": data["lat_max"],
            "inter_max": data["inter_max"]
        },
        "resources": {
            "bram": data["bram"],
            "dsp": data["dsp"],
            "ff": data["ff"],
            "lut": data["lut"]
        },
        "goals": ["fix errors"]
    }
    if goals is not None:
        feedback["goals"].extend(goals)

    if feedback["latency"]["max"] == 0:
        feedback["errors"].append("The maximum latency is 0, which may indicate an issue with the synthesis process.")
    return feedback

# Function to parse through the HLS report and extract the relevant data
def parse_synth_report(file_path):
    data = {
        "Fmax": None,
        "lat_min": None,
        "lat_max": None,
        "inter_max": None,
        "bram": None,
        "dsp": None,
        "ff": None,
        "lut": None
    }

    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    # --- Extract from Timing Summary section ---
    for i, line in enumerate(lines):
        if "+ Timing:" in line:
            # The data row is typically 6 lines after the "+ Timing:" header
            try:
                timing_row = lines[i + 5]
                # Example line: "    |ap_clk  |  10.00 ns|  13.640 ns|     2.70 ns|"
                numbers = re.findall(r"\d+\.\d+", timing_row)
                if len(numbers) >= 2:
                    data["Fmax"] = float(numbers[1])
            except IndexError:
                pass
            break  # Stop after finding the first Latency block
        
        
    # --- Extract from Latency Summary section ---
    for i, line in enumerate(lines):
        if "+ Latency:" in line:
            # The data row is typically 6 lines after the "+ Latency:" header
            try:
                latency_row = lines[i + 6]
                # Example line: "|     3811|     3820|  38.110 us|  38.200 us|  3812|  3821|       no|"
                numbers = re.findall(r"\d+", latency_row)
                if len(numbers) >= 6:
                    data["lat_min"] = int(numbers[0])
                    data["lat_max"] = int(numbers[1])
                    data["inter_max"] = int(numbers[7])
            except IndexError:
                pass
            break  # Stop after finding the first Latency block

    # --- Extract from Utilization Estimates section ---
    for i, line in enumerate(lines):
        if "Utilization Estimates" in line:
            for j in range(i, i + 20):
                if j >= len(lines):
                    break
                if lines[j].strip().startswith("|Total"):
                    util_row = lines[j]
                    numbers = re.findall(r"\d+|-", util_row)
                    try:
                        # Assuming layout: BRAM, dsp, ff, lut, URAM (total 6 cols)
                        data["bram"] = int(numbers[0]) if numbers[0].isdigit() else 0
                        data["dsp"] = int(numbers[1]) if numbers[1].isdigit() else 0
                        data["ff"] = int(numbers[2]) if numbers[2].isdigit() else 0
                        data["lut"] = int(numbers[3]) if numbers[3].isdigit() else 0
                    except IndexError:
                        pass
                    break
            break  # Stop after processing utilization section

    return data

def parse_sim_report(file_path=None):
    feedback = {
        "status": "failed",
        "match": "mismatch",
        "errors": [],
        "warnings": [],
        "expected_v_actual": []
    }
    if file_path is not None:
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
        except FileNotFoundError:
            logger.error(f"The file {file_path} does not exist.")
            return feedback
        
    
    # Regex patterns for different types of errors/warnings
    # Matches: file_path:line:col: (error|warning): message
    compiler_msg_pattern = re.compile(r'^(.*?):(\d+):(\d+):\s+(error|warning):\s+(.*)$')

    for i, line in enumerate(lines):
        compiler_match = compiler_msg_pattern.match(line)
        if compiler_match:
            file_path, line_num, col_num, msg_type, message = compiler_match.groups()
            err_details = {
                'file': file_path.strip(),
                'line': int(line_num),
                'column': int(col_num),
                'message': message.strip()
            }
            if msg_type == "error":
                feedback["errors"].append(err_details)
                feedback["status"] = "failed" # If there's a compilation error, it's failed
            else: # msg_type == "warning"
                feedback["warnings"].append(err_details)
            continue # Move to the next line, as this line is handled


        if "Compilation Status:" in line:
            feedback["status"] = line.split(":")[1].strip()
        elif "SOME TESTS FAILED!" in line:
            feedback["match"] = "mismatch"
        elif "ERROR:" in line:
            feedback["errors"].append(line.split("ERROR:")[1].strip())
        elif "WARNING:" in line:
            feedback["warnings"].append(line.split("WARNING:")[1].strip())
        elif "FAIL:" in line:
            feedback["expected_v_actual"].append(line.split("- ")[1].strip())
            feedback["match"] = "mismatch"
        elif "Percent Error:" in line:
            feedback["percent_error"] = line.split(":")[1].strip()
        elif "ALL TESTS PASSED!" in line:
            feedback["match"] = "match"
            feedback["status"] = "passed"

    
    return feedback

def gather_parsed_data(kernel_name, model="gpt4o", attempt=1, goals=None):
    f""" 
        Send in the  kernel name and the model and attempt and it will search the solutions for the specific
        file where to parse data from.
        
        The goals is optional if you want to send in specific goals for the optimization of the HLS Code. 
        Send the goals as ["new goal", "another new goal"]
    """
    #Create all the files needed if not already created
    solutions_path = f"{file_path_begin}{model}/{kernel_name}/attempts_{attempt}"
    pathlib.Path(solutions_path).mkdir(parents=True, exist_ok=True)
    
    in_report_file = f"{solutions_path}{in_file_path_mid}{kernel_name}{rpt_file_end}"
    log_file_dir = f"{solutions_path}{log_folder_path}"
    pathlib.Path(log_file_dir).mkdir(parents=True, exist_ok=True)
    old_sim_log_file = f"{solutions_path}/logs/hls_run_csim.log"
    
    log_file = f"{log_file_dir}/{kernel_name}_synth{timestamp}.log"

    final_feedback = {
        "csim_feedback": None,
        "synth_feedback": None,
    }
    # Read in the log file from the HLS synthesis attempt
    try:
        with open(f"{solutions_path}{saved_log_file}", "r") as f:
            old_log = f.read()
            with open(log_file, "w") as log:
                log.write(old_log)
                log.write("\n\n")
    except FileNotFoundError:
        logger.error(
            f"The file {solutions_path}{saved_log_file} does not exist."
        )
        old_log = None
    # Check if the synthesis was successful then add the data to json
    try:
        with open(in_report_file, 'r') as file:
            report = file.read()
            summary = parse_synth_report(in_report_file)
            with open(log_file, "a") as log:
                log.write("\n\n")
                log.write(report)
                log.write("\n\n")
                
            synth_feedback = gather_feedback(old_log, summary, goals=goals)

    except FileNotFoundError:
        synth_feedback = gather_feedback(old_log, goals=goals)

        logger.error(
            f"The file {in_report_file} does not exist."
        )
    
    # Check if the simulation was successful then add the data to json
    try:
        with open(old_sim_log_file, 'r') as file:
            report = file.read()
            with open(log_file, "a") as log:
                log.write("\n\n")
                log.write(report)
                log.write("\n\n")
                
            sim_feedback = parse_sim_report(old_sim_log_file)
            # with open(out_data_path, 'w') as of:
            #     json.dump(feedback, of, indent=4)
    except FileNotFoundError:
        sim_feedback = parse_sim_report()
        # with open(out_data_path, 'w') as of:
        #     json.dump(feedback, of, indent=4)
        logger.error(
            f"The file {old_sim_log_file} does not exist."
        )
    
    
    final_feedback["csim_feedback"] = sim_feedback
    final_feedback["synth_feedback"] = synth_feedback    
        
    return final_feedback



def save_feedback_data(out_data_path, feedback = None):
    """Function to save the feedback data to a json file."""
    if feedback is None:
        logger.error("No feedback data provided to save.")
        return
    
    try:
        with open(out_data_path, 'w') as of:
            json.dump(feedback, of, indent=4)
        logger.info(f"Feedback data saved to {out_data_path}")
    except Exception as e:
        logger.error(f"Failed to save feedback data: {e}")

    
# # Example usage
# if __name__ == "__main__":
#     parse_synth_reports()