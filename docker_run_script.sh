#!/bin/bash
#
# Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

docker run -it \
	--device=/dev/accel/accel0:/dev/accel/accel0 \
	--cap-add=NET_ADMIN \
	-v $(pwd):/workspace \
	npueval \
	bash -c "cd /workspace && python3 $1"
