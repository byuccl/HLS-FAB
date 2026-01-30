# Copyright (C) 2025, Advanced Micro Devices, Inc.  All rights reserved.
#
# Author: Preston Walker, AMD
# SPDX-License-Identifier: MIT

.PHONY: all dataset run_completions test parse clean  

# ######### Help Setup #############
# .PHONT: help
# help::
# 	@echo ""
# 	@echo "Makefile Usage:"
# 	@echo " 	This Makefile controls all the HLSFAB flow"
# 	@echo ""
# 	@echo ""
# 	@echo ""
# 	@echo ""
# 	@echo ""
# 	@echo ""


# ########### Setting Variables ###########
# MK_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
# export HLSFAB_PROJECT_PATH := $(patsubst %/,%,$(dir $(MK_PATH)))
# export VITIS_RES_PATH := $(HLSFAB_PROJECT_PATH)/results/solutions/vitis

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

run_vitis_completions: 
	@echo "Running HLSFAB with Vitis"
	python hls_completions.py --hls_flow vitis $(if $(LOG),--log $(LOG),) $(if $(N),--attempts $(N),) $(if $(K), -k $(K),) $(if $(SKIP_BEHAVIORAL), --skip_behavioral) $(if $(RAG), --rag)
run_dynarapid_completions: 
	@echo "Running HLSFAB With Dynarapid"
	python hls_completions.py --hls_flow dynarapid $(if $(LOG),--log $(LOG),) $(if $(N),--attempts $(N),) $(if $(K), -k $(K),) $(if $(RAPIDWRIGHT),--rapidwright)
clean:  
	make -C dataset/ clean  
	rm -rf kernel.json temp.cc  
	rm -rf results/testing/test_results.json  
