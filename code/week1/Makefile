MCC       ?= mcc
CXXFLAGS  ?= -O2 -std=c++17
MUSA_PATH ?= /usr/local/musa-3.1.0
LDFLAGS   ?= -L$(MUSA_PATH)/lib -lmusa -lmusart

BINS = 01_hello_world 02_thread_index 03_device_info \
       04_memory_basics 05_error_check 06_async_kernel

.PHONY: all clean
all: $(BINS)

%: %.mu
	$(MCC) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
