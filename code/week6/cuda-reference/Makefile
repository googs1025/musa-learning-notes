# Week 6 CUDA reference examples: libraries, multi-GPU, and debugging.

BACKEND ?= cuda
NVCC ?= nvcc
MCC ?= mcc
MUSA_HOME ?= /usr/local/musa
CUDA_ARCH ?=
MUSA_ARCH ?=
TARGET ?= all
COMMON_DIR ?= ../../week1/cuda-reference/common
SOURCES := $(shell find . -type f -name '*.cu' -not -path './build/*' | sort)
target_name = $(subst /,__,$(patsubst ./%,%,$(patsubst %.cu,%,$(1))))
TARGETS := $(foreach source,$(SOURCES),$(call target_name,$(source)))

# Extra libraries, multiple devices, MPI, debugger support, or unsafe behavior.
OPTIONAL_TARGETS := chapter08__cublas chapter08__cusparse chapter08__cufft \
  chapter09__simpleP2P_PingPong \
  chapter10__debug-hazards chapter10__debug-segfault chapter10__debug-segfault.fixed \
  chapter10__crypt.parallelized chapter10__crypt.overlap
DEFAULT_TARGETS := $(filter-out $(OPTIONAL_TARGETS),$(TARGETS))

ifeq ($(BACKEND),cuda)
  COMPILER := $(NVCC)
  BACKEND_FLAGS := -O2 -lineinfo $(if $(CUDA_ARCH),-arch=$(CUDA_ARCH),)
  BACKEND_DEFINES :=
else ifeq ($(BACKEND),musa)
  COMPILER := $(MCC)
  BACKEND_FLAGS := -O2 -lineinfo --musa-path=$(MUSA_HOME) -mtgpu $(if $(MUSA_ARCH),--offload-arch=$(MUSA_ARCH),)
  BACKEND_DEFINES := -DCUDA_REFERENCE_MUSA
else
  $(error Unknown BACKEND '$(BACKEND)'; choose cuda or musa)
endif

ifneq ($(TARGET),all)
  ifeq ($(filter $(TARGET),$(TARGETS)),)
    $(error Unknown TARGET '$(TARGET)'; available targets: all$(if $(TARGETS), $(TARGETS), none yet))
  endif
endif

.DEFAULT_GOAL := $(if $(filter all,$(TARGET)),all,$(TARGET))
.PHONY: all clean optional

define make-target
$(call target_name,$(1)): $(1)
	@mkdir -p build
	$(COMPILER) $(BACKEND_FLAGS) $(BACKEND_DEFINES) -I. -I$(COMMON_DIR) $$< -o build/$(call target_name,$(1))
endef
$(foreach source,$(SOURCES),$(eval $(call make-target,$(source))))

ifeq ($(strip $(TARGETS)),)
all:
	@echo "$(COMPILER) $(BACKEND_FLAGS) <future-target>.cu -o <future-target>"
	@echo "No reference source targets imported yet; this is a dry-run build skeleton."
else
all: $(DEFAULT_TARGETS)
endif

optional: $(OPTIONAL_TARGETS)

clean:
	@rm -rf ./build
