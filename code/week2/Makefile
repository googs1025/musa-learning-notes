MCC       ?= mcc
CXXFLAGS  ?= -O2 -std=c++17
MUSA_PATH ?= /usr/local/musa
INCLUDES  ?= -I$(MUSA_PATH)/include -I../include
LDFLAGS   ?= -L$(MUSA_PATH)/lib -lmusa -lmusart

BINS = 01_vector_add_runtime 02_vector_add_pinned 03_vector_add_timer \
       04_vector_add_unified 05_multi_stream 06_stream_event_dep \
       07_musa_graph 08_stream_callback

.PHONY: all clean
all: $(BINS)

%: %.mu
	$(MCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
