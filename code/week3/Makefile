MTCC      ?= mtcc
CXXFLAGS  ?= -O2 -std=c++17
MUSA_PATH ?= /usr/local/musa
INCLUDES  ?= -I$(MUSA_PATH)/include -I../include
LDFLAGS   ?= -L$(MUSA_PATH)/lib -lmusa -lmusart

BINS = 01_warp_divergence 02_reduce_naive 03_reduce_unrolling \
       04_reduce_shfl 05_nested_hello 06_sum_matrix_2d

.PHONY: all clean
all: $(BINS)

%: %.mu
	$(MTCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
