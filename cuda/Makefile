CUDA_DIR = /usr/local/cuda

CC_SRCS := $(wildcard src/*.cc)
CC_OBJS := ${CC_SRCS:src/%.cc=build/obj/%.o}
CUDA_SRCS := $(wildcard src/*.cu)
CUDA_OBJS := ${CUDA_SRCS:src/%.cu=build/obj/%.o}
OBJS := $(CC_OBJS) $(CUDA_OBJS)

CC = g++
WARNINGS = -Wall -Wfatal-errors -Wno-unused -Wno-unused-result
CC_FLAGS = -std=c++11 -fPIC $(WARNINGS) -I$(CUDA_DIR)/include
LD_FLAGS = -L$(CUDA_DIR)/lib64 -lcuda -lcudart -lcublas

NVCC = nvcc
NVCC_FLAGS = -std=c++11 --compiler-options '-fPIC'
ARCH = -gencode arch=compute_30,code=sm_30 \
       -gencode arch=compute_35,code=sm_35 \
       -gencode arch=compute_50,code=[sm_50,compute_50] \
       -gencode arch=compute_52,code=[sm_52,compute_52]

all: build/lib/libc_runtime_api.so

build/lib/libc_runtime_api.so: $(OBJS)
	@mkdir -p build/lib
	$(CC) -shared $^ -o $@ $(LD_FLAGS)

build/obj/%.o: src/%.cc
	@mkdir -p build/obj
	$(CC) $(CC_FLAGS) -c $< -o $@

build/obj/%.o: src/%.cu
	@mkdir -p build/obj
	$(NVCC) $(ARCH) $(NVCC_FLAGS) -c $< -o $@

clean:
	rm -rf build

.PHONY: clean
