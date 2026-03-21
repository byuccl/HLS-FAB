import json


SYSTEM_PROMPTS = {
    "dynarapid": {
        "skip_behavioral_False": """
You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Dynamatic HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time as possible (latency) 
Make this code so that it is Deterministic when synthesized.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Dynamatic does not support arbitrary precision integers (from the <ap_int.h> library) 
The integer types you are allowed to use are int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Dynarapid does not use or recognize any #pragma HLS lines so do not include them. 

""",
        "skip_behavioral_True": ""
    },
    "vitis": {
        "skip_behavioral_True": """  You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled with Vitis HLS tools to run on an AMD Virtex UltraScale+ FPGA.
The goal is to create HLS code that runs correctly, with the lowest possible latency while meeting the timing constraint of 3.3 ns (fmax). Reducing hardware resources is a secondary goal, but should not be prioritized over latency improvements.
Do not use size as a function argument, instead use a constant defined at the top of the file or function.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Most kernels are dependent on the arbitrary precision integer library for HLS (<ap_int.h>) or the arbitrary precision floating point library (<ap_float.h>). As such, that header almost always needs to be added.
Some data types are represented as these arbitrary precision integers or floating point numbers, for example, a bfloat16 is represented with the type `ap_float<16,8>`.
Keep in mind that arbitrary precision integers and floating point numbers often can't be used with standard C++ integers and floats, casting may be necessary to perform operations between them.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Produce only the kernel function, no main, no additional examples or code.
""",
        "skip_behavioral_False": """You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled with Vitis HLS tools to run on an AMD Virtex UltraScale+ FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Do not use size as a function argument, instead use a constant defined at the top of the file or function.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Most kernels are dependent on the arbitrary precision integer library for HLS (<ap_int.h>) or the arbitrary precision floating point library (<ap_float.h>). As such, that header almost always needs to be added.
Some data types are represented as these arbitrary precision integers or floating point numbers, for example, a bfloat16 is represented with the type `ap_float<16,8>`.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Produce only the kernel function, no main, no additional examples or code.
"""
    },
}


VITIS_SYSTEM_PROMPT = """You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Vitis HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Do not use size as a function argument, instead use a constant defined at the top of the file or function.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Most kernels are dependent on the arbitrary precision integer library for HLS (<ap_int.h>) or the arbitrary precision floating point library (<ap_float.h>). As such, that header almost always needs to be added.
Some data types are represented as these arbitrary precision integers or floating point numbers, for example, a bfloat16 is represented with the type `ap_float<16,8>`.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Produce only the kernel function, no main, no additional examples or code.
"""

VITIS_LIBRARY_SYSTEM_PROMPT = """You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Vitis HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Do not use size as a function argument, instead use a constant defined at the top of the file or function.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Most kernels are dependent on the standard integer library for HLS (<stdint.h>). As such, that header almost always needs to be added.
Some data types are represented as these arbitrary precision integers or floating point numbers, for example, a bfloat16 is represented with the type `ap_float<16,8>`.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Produce only the kernel function, no main, no additional examples or code.
"""

DYNARAPID_SYSTEM_PROMPT = """
You are a part of a code generation system for HLS (High Level Synthesis).
Your job is to write C++ code for a single kernel that will be compiled to with Dynamatic HLS tools to run on an AMD Zynq UltraScale+ MPSoC FPGA.
The goal is to create HLS code that runs correctly, taking as little time and using as few hardware resources as possible. 
Make this code so that it is Deterministic when synthesized.
Produce only the C++ code for the requested kernel including any required headers and imports. 
Dynamatic does not support arbitrary precision integers (from the <ap_int.h> library) 
The integer types you are allowed to use are int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t.
Each kernel description includes the function signature of the kernel. Follow this format exactly to allow the program to compile properly. Do not add extra parameters apart from those in the function signature given.
Make sure the C++ code is complete and self contained in a single code block.
Dynarapid does not use or recognize any #pragma HLS lines so do not include them. 
""",


def feedback_prompt_vitis(prev_code, feedback):
    return f"""
Here is my C HLS code:

{prev_code}

Here is some feedback from the synthesis report and the testbench:

{json.dumps(feedback, indent=2)}

Please modify the code using the goals from the feedback as a guideline.

Prioritize the goals to improve the code in the following order:
TIER 1 - CRITICAL ERRORS (Must Fix First):
- Compilation failures (status: failed)
- Synthesis failures (latency: NULL, resources: NULL, errors: non-empty)  
- Simulation mismatches (percent_error > 0%, match: mismatch)
- Function returns non-zero

TIER 2 - PERFORMANCE WARNINGS (Address if Tier 1 is clear):
- II Violations → Memory port contention
- Resource overflow → Exceeding FPGA limits (BRAM: 720, DSP: 2280, FF: 788160 , LUT: 394080)
- Timing violations → Clock frequency issues

TIER 3 - LATENCY IMPROVEMENT OPPORTUNITIES (Prioritize if no warnings or errors remain):
- Latency reduction

TIER 4 - RESOURCE UTILIZATION IMPROVEMENTS (Improve if possible, but not at the cost of any other improvements)
- Resource utilization

Analyze the C Simulation and C Synthesis feedback to determine whether the code meets the first and second goals before moving on to the third goal of optimization
"""


def feedback_prompt_dyarapid(prev_code, feedback):
    return f"""
Here is my C HLS code:

{prev_code}

Here is some feedback from the simulation and dynarapid run:

{json.dumps(feedback, indent=2)}

Please modify the code with the following Goals in mind. 

Create dot and dcp files:
- Each run should have a kernel_name.dot in its directories. This is necessary in order to create the next step kernel_name_routed.dcp.
- Each run should also have a kernel_name_routed.dcp which is for the implementation stage. If it does not have one then edit the code to correctly create one.

Optimize the logic:
- We always want to make our hardware faster and dynarapid is based on Dataflow logic. Make the code work better for Dataflow. 


"""

