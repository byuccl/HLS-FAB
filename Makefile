.PHONY: all dataset run_completions test parse clean  

all: dataset   

ifeq ($(VITIS),ON)
all: run_vitis_completions

else ifeq ($(DYNARAPID), ON)
all: run_dynarapid_completions
endif

generate_tests:  
	@echo "Generating tests..."  
	python3 dataset/generate_tests.py $(WORKFLOW) $(if $(LOG),--log $(LOG),)  
dataset:  
	@echo "Building kernels dataset..."  
	make -C dataset/ all   

run_vitis_completions: dataset
	@echo "Running HLSEval with Vitis"
	python hls_completions_loop.py $(WORKFLOW) --hls_flow vitis --goals "reduce latency" "minimize warnings" $(if $(LOG),--log $(LOG),) $(if $(ATTEMPTS),--attempts $(ATTEMPTS),) $(if $(K), -k $(K),) 
run_dynarapid_completions: dataset
	@echo "Running HLSEval With Dynarapid"
	python hls_completions_loop.py $(WORKFLOW) --hls_flow dynarapid --goals "reduce latency" "minimize warnings" $(if $(LOG),--log $(LOG),) $(if $(ATTEMPTS),--attempts $(ATTEMPTS),) $(if $(K), -k $(K),) $(if $(RAPIDWRITE),--rapidwrite)
clean:  
	make -C dataset/ clean  
	rm -rf kernel.json temp.cc  
	rm -rf results/testing/test_results.json  
