.PHONY: all dataset run_completions test parse clean  

# ######### Help Setup #############
# .PHONT: help
# help::
# 	@echo ""
# 	@echo "Makefile Usage:"
# 	@echo " 	This Makefile controls all the HLSeval flow"
# 	@echo ""
# 	@echo ""
# 	@echo ""
# 	@echo ""
# 	@echo ""
# 	@echo ""


# ########### Setting Variables ###########
# MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
# export HLSEVAL_PROJECT_PATH := $(patsubst %/,%,$(dir $(MK_PATH)))
# export VITIS_RES_PATH := $(HLSEVAL_PROJECT_PATH)/results/solutions/vitis

# #Include your own absolute path to the blas folder in the Vitis_Libraries
# export XF_PROJ_PATH ?= /home/hoodoo17/Desktop/HLSPractice/VitisLibraries/Vitis_Libraries/blas


all: dataset   

ifeq ($(VITIS),ON)
all: run_vitis_completions

else ifeq ($(DYNARAPID), ON)
all: run_dynarapid_completions
endif

generate_tests:  
	@echo "Generating tests..."  
	python3 dataset/generate_tests.py $(if $(LOG),--log $(LOG),)  
dataset:  
	@echo "Building kernels dataset..."  
	make -C dataset/ all   

run_vitis_completions: dataset
	@echo "Running HLSEval with Vitis"
	python hls_completions.py --hls_flow vitis $(if $(LOG),--log $(LOG),) $(if $(ATTEMPTS),--attempts $(ATTEMPTS),) $(if $(K), -k $(K),) $(if $(SKIP_BEHAVIORAL), --skip_behavioral) $(if $(RAG), --rag)
run_dynarapid_completions: dataset
	@echo "Running HLSEval With Dynarapid"
	python hls_completions.py --hls_flow dynarapid $(if $(LOG),--log $(LOG),) $(if $(ATTEMPTS),--attempts $(ATTEMPTS),) $(if $(K), -k $(K),) $(if $(RAPIDWRITE),--rapidwrite)
clean:  
	make -C dataset/ clean  
	rm -rf kernel.json temp.cc  
	rm -rf results/testing/test_results.json  
