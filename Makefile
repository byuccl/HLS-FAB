.PHONY: all dataset run_completions test parse clean  
all:   
	make dataset   
	make run_vitis_completions
generate_tests:  
	@echo "Generating tests..."  
	python3 dataset/generate_tests.py $(WORKFLOW) $(if $(LOG),--log $(LOG),)  
dataset:  
	@echo "Building kernels dataset..."  
	make -C dataset/ all   
run_vitis_completions:   
	@echo "\n\nRunning HLS completions..."  
	python hls_completions_loop.py $(WORKFLOW) --hls_flow vitis --goals "reduce latency" "minimize warnings" $(if $(LOG),--log $(LOG),) $(if $(ATTEMPTS),--attempts $(ATTEMPTS),)  
run_dynarapid_completions:
	python hls_completions_loop.py $(WORKFLOW) --hls_flow dynarapid --goals "reduce latency" "minimize warnings" $(if $(LOG),--log $(LOG),) $(if $(ATTEMPTS),--attempts $(ATTEMPTS),) $(if $(RAPIDWRITE),--rapidwrite)
test:   
	@echo "\n\nRunning tests..."  
	python3 results/testing/run_tests.py  
  
clean:  
	make -C dataset/ clean  
	rm -rf kernel.json temp.cc  
	rm -rf results/testing/test_results.json  
