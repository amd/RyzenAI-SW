FLEXML_XTC ?= /proj/testcases/xtc/HEAD/tc/open/flexml
MODEL_SOURCE_PATH := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))

ifeq ($(shell test -d $(MODEL_SOURCE_PATH)/../../../common && echo -n yes),yes)
	PERFORCE_HOME = $(MODEL_SOURCE_PATH)/../../..
else
	PERFORCE_HOME = $(FLEXML_XTC)
endif

ifeq ($(shell test -d $(MODEL_SOURCE_PATH)/../../common && echo -n yes),yes)
	PERFORCE_E2E_HOME = $(MODEL_SOURCE_PATH)/../../
else
	PERFORCE_E2E_HOME = $(FLEXML_XTC)/e2e-hw
endif

MODEL = ResNet18
INPUT_MODEL = ResNet18.onnx
TENSOR_SHAPE = 1x3x224x224
VAIMLEP_RUN_SCRIPT = run.py
DEVICE = stx
FLEXML_USE_STD_SYSTEM_CALL = 1

include $(PERFORCE_E2E_HOME)/common/Makefile-common.mk
