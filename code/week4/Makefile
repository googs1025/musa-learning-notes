MCC      ?= mcc
CXXFLAGS  ?= -O2 -std=c++17
MUSA_PATH ?= /usr/local/musa
INCLUDES  ?= -I$(MUSA_PATH)/include -I../include
LDFLAGS   ?= -L$(MUSA_PATH)/lib -lmusa -lmusart

BINS = 01_saxpy_bandwidth 02_offset_access 03_offset_unrolling \
       04_aos_vs_soa 05_transpose_naive 06_transpose_padded

.PHONY: all clean
all: $(BINS)

%: %.mu
	$(MCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
