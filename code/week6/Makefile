MCC      ?= mcc
CXX       ?= g++
CXXFLAGS  ?= -O2 -std=c++17
MUSA_PATH ?= /usr/local/musa
INCLUDES  ?= -I$(MUSA_PATH)/include -I../include
LDFLAGS   ?= -L$(MUSA_PATH)/lib -lmusa -lmusart

BINS = 02_musa_gdb_demo 03_error_dump

.PHONY: all mccl custom-op clean
all: $(BINS)

%: %.mu
	$(MCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

mccl: 01_mccl_allreduce
01_mccl_allreduce: 01_mccl_allreduce.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) -lmccl

custom-op: 05_torch_musa_custom_op
05_torch_musa_custom_op: 05_torch_musa_custom_op.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS) 01_mccl_allreduce 05_torch_musa_custom_op
