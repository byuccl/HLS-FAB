import re
import pandas as pd
import os
from collections import defaultdict

# --- Configuration ---
# 1. Define the cost structure for your models (Price per 1 Million Tokens)
LLM_COSTS = {
    'anthropic/claude-sonnet-4.5':      {'input': 3.00, 'output': 15.00},
    'deepseek/deepseek-chat-v3-0324':   {'input': 0.24, 'output': 0.84},
    'deepseek/deepseek-r1-0528':        {'input': 0.40, 'output': 1.75},
    'google/gemini-2.5-flash':          {'input': 0.30, 'output': 2.50},
    'google/gemini-2.5-pro':            {'input': 1.25, 'output': 10.00},
    'gpt-4.1-mini-2025-04-14':          {'input': 0.40, 'output': 1.60},
    'gpt-5-2025-08-07':                 {'input': 1.25, 'output': 10.00},
    'qwen/qwen3-coder':                 {'input': 0.22, 'output': 0.95},
    'x-ai/grok-4':                      {'input': 3.00, 'output': 15.00},
    'x-ai/grok-code-fast-1':            {'input': 0.20, 'output': 1.50},
}

# 2. Define the REGEX patterns to capture data
API_RESPONSE_PATTERN = re.compile(
    # Capture the unique ID
    r'"id":\s*"(?P<call_id>[^"]+)",.*?' 
    # Capture the model name
    r'"model":\s*"(?P<model_name>[^"]+)",.*?' 
    # Capture the prompt/input tokens
    r'"prompt_tokens":\s*(?P<input_tokens>\d+),.*?'
    # Capture the completion/output tokens
    r'"completion_tokens":\s*(?P<output_tokens>\d+)',
    re.DOTALL | re.IGNORECASE # DOTALL handles multi-line responses
)
# 3. Dynamically load all log files from the results/logs directory
LOG_DIR = "results/logs"
LOG_FILES = [os.path.join(LOG_DIR, f) for f in os.listdir(LOG_DIR) if f.endswith('.log')]


# --- Data Storage ---
# Stores temporary data: call_id -> {'model', 'input', 'output'}
call_data = defaultdict(dict)
all_records = []


# ====================================================================
# A. LOG FILE PARSING (Data Gathering)
# ====================================================================

def parse_files(file_list):
    """Iterates through files and extracts Model and Token data using regex."""

    all_records = []
    for filename in file_list:
        if not os.path.exists(filename):
            print(f"Warning: File not found: {filename}")
            continue
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            print(process_log_lines(lines))
            all_records.extend(process_log_lines(lines)) # Appends a list of all the model calls with input and output tokens associated with it

    return all_records

# Assuming you have a list of all lines from a single log file
def process_log_lines(lines):
    all_records = []
    
    # 1. Join all lines into a single string for multi-line regex search
    log_content = "".join(lines)
    
    # 2. Iterate over all matches of the full API block
    for match in API_RESPONSE_PATTERN.finditer(log_content):
        data = match.groupdict()
        
        # Add the complete record to the list
        all_records.append({
            'model': data['model_name'],
            'input_tokens': int(data['input_tokens']),
            'output_tokens': int(data['output_tokens'])
        })
    
    return all_records


# Execute the parsing function
all_records = parse_files(LOG_FILES)
# ====================================================================
# B. DATA CLEANUP AND PREPARATION
# ====================================================================

# Consolidate linked data into a list of complete records
# for call_id, data in call_data.items():
#     # A complete record must have a model name and both token counts
#     if all(k in data for k in ['model', 'input_tokens', 'output_tokens']):
#         all_records.append({
#             'model': data['model'],
#             'input_tokens': data['input'],
#             'output_tokens': data['output']
#         })

# Create the master DataFrame
df = pd.DataFrame(all_records)
if df.empty:
    print("Error: No complete LLM call records were found. Exiting.")
    exit()

print(f"Successfully processed {len(df)} complete LLM calls.")


# ====================================================================
# C. CALCULATION AND ANALYSIS
# ====================================================================

# 1. Calculate Average Tokens Per Model
# Use .agg() to get both sum and count, then calculate the mean
model_summary = df.groupby('model').agg(
    total_input_tokens=('input_tokens', 'sum'),
    total_output_tokens=('output_tokens', 'sum'),
    call_count=('model', 'count'),
    std_input_tokens=('input_tokens', 'std'),
    std_output_tokens=('output_tokens', 'std')
).reset_index()

# Calculate Averages
model_summary['avg_input_tokens'] = model_summary['total_input_tokens'] / model_summary['call_count']
model_summary['avg_output_tokens'] = model_summary['total_output_tokens'] / model_summary['call_count']


# 2. Calculate cost per individual call and add to main dataframe
def calculate_cost_per_call(model, input_tokens, output_tokens):
    """Calculates the cost for a single API call."""
    input_cost_per_m = LLM_COSTS.get(model, {}).get('input', 0)
    output_cost_per_m = LLM_COSTS.get(model, {}).get('output', 0)
    
    # Calculate costs (token count * cost per 1M tokens / 1M)
    input_cost = input_tokens * (input_cost_per_m / 1_000_000)
    output_cost = output_tokens * (output_cost_per_m / 1_000_000)
    
    return input_cost + output_cost

# Add cost per call to the main dataframe
df['cost'] = df.apply(lambda row: calculate_cost_per_call(row['model'], row['input_tokens'], row['output_tokens']), axis=1)

# Calculate Average Cost and Std Dev for each model
cost_stats = df.groupby('model').agg(
    avg_cost=('cost', 'mean'),
    std_cost=('cost', 'std')
).reset_index()

# Merge cost statistics into model_summary
model_summary = model_summary.merge(cost_stats, on='model', how='left')


# 3. Final Output Formatting
# Select and reorder columns, round to 2 decimal places for currency
final_df = model_summary[[
    'model', 
    'call_count', 
    'avg_input_tokens',
    'std_input_tokens',
    'avg_output_tokens',
    'std_output_tokens',
    'avg_cost',
    'std_cost'
]]

# Final presentation of the results
print("\n=======================================================")
print("           LLM COST AND USAGE SUMMARY")
print("=======================================================")

# Print with dollar sign formatting for cost
print(final_df.round(6).to_markdown(index=False, numalign="left", stralign="left", floatfmt=".4f"))