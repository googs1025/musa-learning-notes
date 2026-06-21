MTCC      ?= mtcc
CXXFLAGS  ?= -O2 -std=c++17
MUSA_PATH ?= /usr/local/musa
INCLUDES  ?= -I$(MUSA_PATH)/include -I../include
LDFLAGS   ?= -L$(MUSA_PATH)/lib -lmusa -lmusart

BINS = 01_shared_basics 02_reduce_shared 03_transpose_shared \
       04_stencil_constant 05_naive_gemm 06_tiled_gemm
OPTIONAL_BINS = 07_mublas_sgemm

.PHONY: all optional clean
all: $(BINS)
optional: $(OPTIONAL_BINS)

%: %.mu
	$(MTCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

07_mublas_sgemm: 07_mublas_sgemm.mu
	$(MTCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS) -lmublas

clean:
	rm -f $(BINS) $(OPTIONAL_BINS)
