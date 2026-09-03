MCC       ?= mcc
CXXFLAGS  ?= -O2 -std=c++17
MUSA_PATH ?= /usr/local/musa
INCLUDES  ?= -I$(MUSA_PATH)/include -I../include
LDFLAGS   ?= -L$(MUSA_PATH)/lib -lmusa -lmusart

BINS = 01_row_softmax 02_online_softmax 03_naive_attention \
       04_fused_attention_small_d 05_flash_attention_mini

.PHONY: all clean
all: $(BINS)

%: %.mu
	$(MCC) $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(BINS)
