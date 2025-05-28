.PHONY: all kernels dataset clean

all: kernels dataset

kernels:
	./generate_kernels.sh

dataset:
	python3 generate_hlseval.py
tests: 
	python3 generate_tests.py $(WORKFLOW)
clean:
	rm -f hlseval.jsonl
	rm -f kernels/*/kernel.json
	rm -rf kernels/*/__pycache__/
	rm -f kernel_list.txt
	rm -f kernel_testbench_includes.h
	rm -f test_function_map.h
