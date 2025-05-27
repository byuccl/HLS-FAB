.PHONY: all dataset run_completions test parse clean
all: 
	make dataset 
	make run_completions_loop
	make test 
generate_tests:
	@echo "Generating tests..."
	python3 dataset/generate_tests.py $(WORKFLOW)
dataset:
	@echo "Building kernels dataset..."
	make -C dataset/ all
run_completions:
	python hls_completions.py $(WORKFLOW)
run_completions_loop: 
	@echo "\n\nRunning HLS completions..."
	python hls_completions_loop.py $(WORKFLOW) --goals "reduce latency" "minimize warnings"
test: 
	@echo "\n\nRunning tests..."
	python3 results/testing/run_tests.py

clean:
	make -C dataset/ clean
	rm -rf kernel.json temp.cc
	rm -rf results/testing/test_results.json