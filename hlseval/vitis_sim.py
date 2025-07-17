#!/usr/bin/env python3

import re
import os
import argparse
import sys

def extract_kernel_info(cpp_file):
    """Extract kernel name, inputs, and outputs from C++ testbench"""
    kernel_info = {
        'name': None,
        'inputs': [],
        'outputs': [],
        'array_sizes': {}
    }
    
    with open(cpp_file, 'r') as f:
        content = f.read()
    
    # Find kernel name
    kernel_pattern = r'#include\s+"([^"]+)\.h"'
    kernel_match = re.search(kernel_pattern, content)
    if kernel_match:
        kernel_info['name'] = kernel_match.group(1)
    
    # Look for function call to extract parameters
    if kernel_info['name']:
        call_pattern = r'\b{}\s*\((.*?)\);'.format(kernel_info['name'])
        call_match = re.search(call_pattern, content, re.DOTALL)
        
        if call_match:
            params = call_match.group(1).split(',')
            for param in params:
                param = param.strip()
                if param.endswith('_arr'):
                    var_name = param[:-4]  # Remove '_arr' suffix
                    
                    # Find array size
                    size_pattern = r'{}_arr\s*=\s*new\s+\w+\s*\[(\d+)\]'.format(re.escape(var_name))
                    size_match = re.search(size_pattern, content)
                    if size_match:
                        size = int(size_match.group(1))
                        kernel_info['array_sizes'][var_name] = size
                    
                    # Determine if input or output
                    if 'input' in var_name.lower():
                        kernel_info['inputs'].append(var_name)
                    elif 'output' in var_name.lower():
                        kernel_info['outputs'].append(var_name)
    
    return kernel_info

def determine_data_type(cpp_file):
    """Determine the data type of arrays used in the C++ testbench"""
    data_types = {}
    
    with open(cpp_file, 'r') as f:
        content = f.read()
    
    # Look for array declarations
    pattern = r'(\w+)_arr\s*=\s*new\s+(\w+)\s*\['
    matches = re.findall(pattern, content)
    
    for var_name, data_type in matches:
        data_types[var_name] = data_type
    
    return data_types

def generate_fast_testbench(kernel_info, data_types, output_file):
    """Generate a fast Verilog testbench for the specified kernel"""
    kernel_name = kernel_info['name']
    inputs = kernel_info['inputs']
    outputs = kernel_info['outputs']
    
    # Determine bit width based on data types
    bit_widths = {}
    for var, dtype in data_types.items():
        if dtype == 'int8_t' or dtype == 'uint8_t':
            bit_widths[var] = 8
        elif dtype == 'int16_t' or dtype == 'uint16_t':
            bit_widths[var] = 16
        elif dtype == 'int32_t' or dtype == 'uint32_t' or dtype == 'float':
            bit_widths[var] = 32
        elif dtype == 'int64_t' or dtype == 'uint64_t' or dtype == 'double':
            bit_widths[var] = 64
        else:
            bit_widths[var] = 32  # Default
    
    # Start generating the testbench
    verilog_code = f"""
`timescale 1ns / 1ps

module {kernel_name}_tb();
    // Clock and reset - Fast clock for simulation
    reg clk = 0;
    reg rst = 1;
    always #0.5 clk = ~clk; // 1GHz simulation clock
    
"""

    # Generate memory interface signals
    for inp in inputs:
        bit_width = bit_widths.get(inp, 32)
        verilog_code += f"""
    // Memory interface for {inp}
    wire [31:0] MC{inp}_address0_Port;
    wire MC{inp}_we0_ce0_Port;
    wire [31:0] MC{inp}_dout0_Port;
    reg [31:0] {inp}_data_reg = 0;
    assign MC{inp}_dout0_Port = {inp}_data_reg;
"""
    
    for out in outputs:
        bit_width = bit_widths.get(out, 32)
        verilog_code += f"""
    // Memory interface for {out}
    wire [31:0] MC{out}_address0_Port;
    wire MC{out}_we0_ce0_Port;
    wire [31:0] MC{out}_din1_Port;
"""
    
    # Generate control signals
    verilog_code += """
    // Control signals
    reg start_valid = 0;
    wire start_ready;
    reg end_ready = 0;
    wire end_valid;
    
    // Test data storage - Small test set for speed
    localparam TEST_SIZE = 4; // Use just 4 samples for quick testing
    integer i, errors;
    
"""

    # Generate test data arrays
    for inp in inputs:
        bit_width = bit_widths.get(inp, 32)
        size = kernel_info['array_sizes'].get(inp, 256)
        verilog_code += f"""
    reg [{bit_width-1}:0] {inp}_test [0:TEST_SIZE-1];
    reg [{bit_width-1}:0] {inp}_mem [0:31]; // Small memory for speed
"""
    
    for out in outputs:
        bit_width = bit_widths.get(out, 32)
        size = kernel_info['array_sizes'].get(out, 256)
        verilog_code += f"""
    reg [{bit_width-1}:0] {out}_expected [0:TEST_SIZE-1];
    reg [{bit_width-1}:0] {out}_mem [0:31]; // Small memory for speed
"""
    
    # Generate DUT instantiation
    verilog_code += f"""
    // Instantiate the DUT
    {kernel_name} DUT (
        .clkin(clk),
        .rst(rst),
        .start0_valid_in_0(start_valid),
        .start0_ready_out_0(start_ready),
        .end_valid_out_0(end_valid),
        .end_ready_in_0(end_ready),
"""

    # Connect input memory ports
    for inp in inputs:
        verilog_code += f"""
        .MC{inp}_address0_Port(MC{inp}_address0_Port),
        .MC{inp}_we0_ce0_Port(MC{inp}_we0_ce0_Port),
        .MC{inp}_dout0_Port(MC{inp}_dout0_Port),
        .MC{inp}_ready_in_1(1'b1),
        .MC{inp}_pValidArray_1_Port(1'b1),
        .MC{inp}_pValidArray_2_Port(1'b1),
        .MC{inp}_pValidArray_3_Port(1'b1),
        .MC{inp}_dataInArray_1_Port(32'h0),
        .MC{inp}_dataInArray_2_Port(32'h0),
        .MC{inp}_dataInArray_3_Port(32'h0),
"""
    
    # Connect output memory ports
    for out in outputs:
        verilog_code += f"""
        .MC{out}_address0_Port(MC{out}_address0_Port),
        .MC{out}_we0_ce0_Port(MC{out}_we0_ce0_Port),
        .MC{out}_din1_Port(MC{out}_din1_Port),
        .MC{out}_ready_in_0(1'b1),
        .MC{out}_nReadyArray_1_Port(1'b1),
        .MC{out}_pValidArray_3_Port(1'b1),
        .MC{out}_dataInArray_3_Port(32'h0),
"""

    # Complete the DUT instantiation
    verilog_code += """
        .icmp9_din_1(32'h0),
        .icmp9_valid_in_1(1'b0),
        .source2_ready_in_0(1'b1)
    );
    
    // Test sequence
    initial begin
"""

    # Generate default test data
    # For simplicity, we'll use simple test patterns
    for inp in inputs:
        bit_width = bit_widths.get(inp, 32)
        verilog_code += f"""
        // Initialize {inp} test data with simple values
        {inp}_test[0] = {bit_width}'h1;
        {inp}_test[1] = {bit_width}'h0;
        {inp}_test[2] = {bit_width}'h{2**(bit_width-1):x}; // Largest negative value
        {inp}_test[3] = {bit_width}'h{2**(bit_width-1)-1:x}; // Largest positive value
        
        // Copy to memory
        for (i = 0; i < TEST_SIZE; i = i + 1)
            {inp}_mem[i] = {inp}_test[i];
"""
    
    for out in outputs:
        bit_width = bit_widths.get(out, 32)
        # For a negate function, but users should update this for other kernels
        verilog_code += f"""
        // Set expected outputs (MODIFY THIS FOR YOUR SPECIFIC KERNEL)
        {out}_expected[0] = -{inp}_test[0]; // Example: negation
        {out}_expected[1] = -{inp}_test[1];
        {out}_expected[2] = -{inp}_test[2];
        {out}_expected[3] = -{inp}_test[3];
"""
    
    # Generate test execution
    verilog_code += """
        // Fast reset
        rst = 1; #1; rst = 0; #1;
        
        // Start execution
        $display("Starting test...");
        start_valid = 1;
        
        // Short wait for handshake
        #2; start_valid = 0;
        
        // Fixed wait time instead of waiting for completion signal
        #500;
        
        // Verify results
        errors = 0;
"""

    # Generate verification code
    for out in outputs:
        bit_width = bit_widths.get(out, 32)
        hex_width = (bit_width + 3) // 4  # Number of hex digits needed
        verilog_code += f"""
        // Check {out} results
        for (i = 0; i < TEST_SIZE; i = i + 1) begin
            if ({out}_mem[i] !== {out}_expected[i]) begin
                $display("Error at index %d: Input=%h, Expected=%h, Got=%h", 
                        i, {inp}_test[i], {out}_expected[i], {out}_mem[i]);
                errors = errors + 1;
            end
        end
"""
    
    # Complete the test
    verilog_code += """
        // Report results
        if (errors == 0)
            $display("TEST PASSED! All test values were correctly processed.");
        else
            $display("TEST FAILED! %d errors out of %d tests.", errors, TEST_SIZE);
            
        $finish;
    end
    
    // Memory operations
    always @(posedge clk) begin
"""

    # Generate memory read/write operations
    for inp in inputs:
        bit_width = bit_widths.get(inp, 32)
        verilog_code += f"""
        // {inp} read operation
        if (MC{inp}_we0_ce0_Port)
            {inp}_data_reg <= {{{{32-{bit_width}{{1'b0}}}}, {inp}_mem[MC{inp}_address0_Port[4:0]]}};
"""
    
    for out in outputs:
        bit_width = bit_widths.get(out, 32)
        verilog_code += f"""
        // {out} write operation
        if (MC{out}_we0_ce0_Port)
            {out}_mem[MC{out}_address0_Port[4:0]] <= MC{out}_din1_Port[{bit_width-1}:0];
"""

    verilog_code += """
    end
endmodule
"""
    
    # Write the testbench to file
    with open(output_file, 'w') as f:
        f.write(verilog_code)
    
    print(f"Generated fast testbench for {kernel_name} in {output_file}")
    return kernel_name

def generate_tcl_script(kernel_name, testbench_path, dcp_path, output_dir):
    """Generate TCL script to run the simulation"""
    tcl_path = os.path.join(output_dir, "run_simulation.tcl")
    
    tcl_script = f"""
# Close any open project
close_project -quiet

# Create project from checkpoint with netlist extraction
open_checkpoint {dcp_path}

# Generate a functional simulation netlist
set netlist_path "{output_dir}/{kernel_name}_netlist.v"
write_verilog -force -mode funcsim $netlist_path

# Create a lightweight project
set project_dir "{output_dir}/vivado_quick_sim"
file mkdir $project_dir
save_project_as quick_sim $project_dir -force

# Add netlist and testbench
add_files -fileset sim_1 $netlist_path
add_files -fileset sim_1 {testbench_path}

# Configure for fastest possible simulation
set_property top {kernel_name}_tb [get_filesets sim_1]
set_property top_lib xil_defaultlib [get_filesets sim_1]
set_property NETLIST_TYPE "FUNCTIONAL" [get_filesets sim_1]
set_property SIMULATION_FLOW "BEHAVIORAL" [get_filesets sim_1]

# Disable all timing checks and annotated delays
set_property -name {{xsim.simulate.type}} -value {{functional}} -objects [get_filesets sim_1]
set_property -name {{xsim.elaborate.sdf_delay}} -value {{none}} -objects [get_filesets sim_1]
set_property -name {{xsim.elaborate.mt_level}} -value {{off}} -objects [get_filesets sim_1]
set_property -name {{xsim.simulate.runtime}} -value {{1000ns}} -objects [get_filesets sim_1]
set_property -name {{xsim.compile.xvlog.nosort}} -value {{true}} -objects [get_filesets sim_1]
set_property -name {{xsim.compile.xvlog.relax}} -value {{true}} -objects [get_filesets sim_1]

# Launch fast simulation
launch_simulation -simset sim_1 -mode behavioral

# Run for minimal time
run 1000ns
"""
    
    with open(tcl_path, 'w') as f:
        f.write(tcl_script)
    
    print(f"Generated TCL script in {tcl_path}")

def main():
    """Main function to handle command-line arguments and generate testbench"""
    parser = argparse.ArgumentParser(description='Generate Verilog testbench from C++ testbench')
    parser.add_argument('cpp_testbench', help='Path to C++ testbench file')
    parser.add_argument('--output', '-o', help='Output directory for generated files', default='.')
    parser.add_argument('--dcp', help='Path to DCP file for simulation', required=True)
    parser.add_argument('--tb-name', help='Custom testbench filename (optional)')
    
    args = parser.parse_args()
    
    # Extract information from C++ testbench
    kernel_info = extract_kernel_info(args.cpp_testbench)
    data_types = determine_data_type(args.cpp_testbench)
    
    if not kernel_info['name']:
        print("Error: Could not extract kernel name from C++ testbench")
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    
    # Determine testbench filename
    if args.tb_name:
        tb_filename = args.tb_name
        if not tb_filename.endswith('.v'):
            tb_filename += '.v'
    else:
        tb_filename = f"{kernel_info['name']}_tb.v"
    
    tb_path = os.path.join(args.output, tb_filename)
    
    # Generate testbench
    kernel_name = generate_fast_testbench(kernel_info, data_types, tb_path)
    
    # Generate TCL script
    generate_tcl_script(kernel_name, tb_path, args.dcp, args.output)
    
    print(f"Generation complete for {kernel_name}.")
    print(f"To run simulation: vivado -mode batch -source {os.path.join(args.output, 'run_simulation.tcl')}")

if __name__ == "__main__":
    main()
