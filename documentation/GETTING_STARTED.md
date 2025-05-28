# Getting Started with HLSEval

	Welcome to HLSEval this document is to help you get started and familiarized with all the tools and resources needed to work on HLSEval with us. 
	There are many bootcamp modules that give a crashcourse on things like python, git and github, and HLS. There are also many resources that are important to understanding the nature of the project for example HumanEval which is similar to this project with a different focus.

## [BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/)
- Contains tutorials for software principles and FPGA development. Go through these and try out the challenges associated. 
### Must Have Skills (feel free to skip if you are comfortable with topic)
- [Linux Setup & Tutorials • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/linux/)
- [Git • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/git/)
- [GitHub • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/github/)
- [Python Intro • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/pythonIntro/)
- [Python Environments • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/pythonEnvs/)
- [Vitis: HLS • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/vitis_hls/)
### Additional Skills
- [Docker • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/docker/)
- [VS Code • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/vscode/)
- [Markdown • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/markdown/)

## HLSEval Background Information
### HumanEval
	**Brief overview:**
	HumanEval was a study done by openai to test their Large Language Models (LLM) ability to write Python code. They created Codex which is specially trained on opensource python code from Github then they compared it against gpt-3 and gpt-J. They wanted to improve the models ability to produce functionally correct code. This is important because we are also doing similar stuf with HLSEval. They use 164 different hand written tests with unit tests that are automatically evaluated. We want to do the same.
- [HumanEval Paper](https://arxiv.org/abs/2107.03374)
- [HumanEval Repository](https://github.com/openai/human-eval)

### HLS
	**Brief Overview:**
	HLS Stand for High Level Synthesis. It is a way to create higher level code for HDL and digital designs. Usually the code is C that is then optimized using #pragma directives for HLS specifcally. **This is the main topic you need to familiarize yourself with to begin.** We are evaluating the effecitveness and then trying to improve the models ability to write this HLS C code. 
- [Vitis: HLS • BYU Computing Bootcamp](https://byu-cpe.github.io/ComputingBootCamp/tutorials/vitis_hls/)
- [High-level synthesis - Wikipedia](https://en.wikipedia.org/wiki/High-level_synthesis)
- [Vitis HLS Guide](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Introduction)
- [ECEn 625 Class Labs](https://byu-cpe.github.io/ecen625/labs/lab-vitis-hls/) *Start with Lab 4 and then do Lab 5.*

### What is HLSEval?
- A project similar to HumanEval, but focused on benchmarking the capability of LLMs to write HLS code and then improve it.
	- NPUEval is a similar project, focused on benchmarking LLMs ability to write code to target AMD's AI cores (Neural Processing Units?)
		- NPUEval hasn't been published yet, but the code for it is contained in the HLSEval repository
- Sponsored by [AMD](https://www.amd.com/en.html)
	- AMD Aquired [Xilinx](https://en.wikipedia.org/wiki/Xilinx) in 2022
	- Vitis HLS tools and many FPGAs are built by AMD
- [HLSEval Repository](https://github.com/skalade/hlseval)
	- Ask someone for access to the github repo. 
## HLSEval HW/SW Requirements
### Hardware (HW) Requirements
- HLS Synthesis is being done for [Zynq UltraScale+ MPSoCs](https://www.amd.com/en/products/adaptive-socs-and-fpgas/soc/zynq-ultrascale-plus-mpsoc.html)
	- Not currently configured to produce a bitstream and run on the board
### Software (SW) Requirements
#### [Vitis Unified Software Platform](https://www.amd.com/en/products/software/adaptive-socs-and-fpgas/vitis.html)
- Vitis is the software with tools to compile, synthesize, etc. the HLS code
- The CCL has a [repository](https://github.com/byuccl/ccl/blob/main/docs/vivado.md "https://github.com/byuccl/ccl/blob/main/docs/vivado.md") for making Vitis and Vivado installation easier.
- After looking at the Bootcamp for the Vitis HLS Start to look at The [AMD Documentation](https://docs.amd.com/r/en-US/ug1399-vitis-hls/Introduction-to-Vitis-HLS-Components) on Vitis and creating Components from C code
- Soon in the project we will use other HLS software tools like Dynamatic and Dynarapid but you do not need to worry about that yet.
#### Python Environment
- The following packages need to be installed either globally or in a venv (recommended):
	- CppHeaderParser
	- openai
	- numpy
## Other Skills
### Using APIs
	In our python environments we use different APIs for running the LLMs. Here are some videos to learn how to run those. 
- [Claude API](https://www.youtube.com/watch?v=W68VEFHCsLc)
- [OpenAI API](https://www.youtube.com/watch?v=YVFWBJ1WVF8)

### Prompt Engineering
	Prompt engineering bascially is the process of improving prompts to generative AI models. We use OpenAI API key. We do a lot of prompting through this API key so learn some of the skills about prompt engineering.
- [Prompt Engineering Guide](https://www.promptingguide.ai/)
## Glossary
* **LLM** Large Language Model
* **Kernel** Basically it is a small snippet of code (in our case HLS code) which can then be used for larged codes. Like specific functions that will be combined together to build larger projects. It is the C/C++ code that can be synthesized into RTL IP Block for Vivado. 
* **RTL** Register Transfer Level: A level of the Digital Design Process between the Higher Level system behavior and the low level hardware implementation.
* **Directives** The #pragmas that tell the HLS software what to specifically do with the code (PIPELINE, INTERFACE, UNROLL, PARTITION)
* **Data Dependencies** Data dependencies are one operation relies on the result or data output from a different operation. Causes bottlenecks in pipelining or limits parallelization of code.
* **Initiation Interval** The interval that a second iteration starts after the previous one in pipelining. (II in the #pragma HLS PIPELINE II=2)
