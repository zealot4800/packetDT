ROOT := $(abspath ..)
BUILD_DIR ?= $(ROOT)/p4/build
MODEL ?= $(ROOT)/results/CIC-IDS-2017-PCAPS1-f10/statedt/model.pkl
PYTHON ?= $(ROOT)/.venv/bin/python
P4C_BMV2 ?= p4c-bm2-ss
P4TEST ?= p4test
BF_P4C ?= p4c-barefoot
VITIS_P4C ?= p4c-vitisnet

.PHONY: generate bmv2 tofino xilinx xilinx-check check-local check

generate:
	$(PYTHON) $(ROOT)/p4/tools/generate_statedt.py $(MODEL) \
		--output-dir $(ROOT)/p4/common

bmv2: generate
	mkdir -p $(BUILD_DIR)/bmv2
	$(P4C_BMV2) --std p4-16 -I $(ROOT)/p4/bmv2 -I $(ROOT)/p4/common \
		-o $(BUILD_DIR)/bmv2/statedt.json $(ROOT)/p4/bmv2/statedt.p4

tofino: generate
	mkdir -p $(BUILD_DIR)/tofino
	$(BF_P4C) --std p4-16 --target tofino --arch tna \
		-I $(ROOT)/p4/tofino -I $(ROOT)/p4/common \
		-o $(BUILD_DIR)/tofino $(ROOT)/p4/tofino/statedt.p4

xilinx: generate
	mkdir -p $(BUILD_DIR)/xilinx
	$(VITIS_P4C) -I $(ROOT)/p4/xilinx -I $(ROOT)/p4/common \
		$(ROOT)/p4/xilinx/statedt.p4 -o $(BUILD_DIR)/xilinx/statedt.json

xilinx-check: generate
	$(P4TEST) --std p4-16 -I $(ROOT)/p4/tests -I $(ROOT)/p4/xilinx \
		-I $(ROOT)/p4/common $(ROOT)/p4/xilinx/statedt.p4

check-local: bmv2 xilinx-check

check: bmv2 tofino xilinx-check
