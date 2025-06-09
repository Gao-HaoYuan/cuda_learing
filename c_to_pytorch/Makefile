AR ?= ar
CXX ?= g++
NVCC ?= nvcc -ccbin $(CXX)
PYTHON ?= python

ifeq ($(OS),Windows_NT)
CUDA_HOME ?= $(CUDA_PATH)
AR := lib
AR_FLAGS := /nologo /out:$(LIBNAME)
NVCC_FLAGS := -x cu -Xcompiler "/MD"
else
CUDA_HOME ?= /usr/local/cuda
AR ?= ar
AR_FLAGS := -crv $(LIBNAME)
NVCC_FLAGS := -std=c++11 -x cu -Xcompiler -fPIC
endif


LOCAL_CFLAGS := -I/usr/include/eigen3 -I$(CUDA_HOME)/include -Ilib -O3
LOCAL_LDFLAGS := -L$(CUDA_HOME)/lib64 -L. -lcudart -lcublas
GPU_ARCH_FLAGS := -gencode=arch=compute_70,code=sm_70 \
                  -gencode=arch=compute_75,code=sm_75 \
                  -gencode=arch=compute_80,code=sm_80 \
                  -gencode=arch=compute_86,code=sm_86 \
                  -gencode=arch=compute_89,code=sm_89 \
                  -gencode=arch=compute_90,code=sm_90 \
                  -gencode=arch=compute_90,code=compute_90

.PHONY: all my_add clean
all: my_add

my_add:
	$(NVCC) $(GPU_ARCH_FLAGS) -c ./my_add/add_kernel.cu -o ./my_add/add_kernel.o $(NVCC_FLAGS) $(LOCAL_CFLAGS)
	$(AR) $(AR_FLAGS) ./my_add/*.o

clean:
	find . \( -iname '*.o' -o -iname '*.so' -o -iname '*.a' -o -iname '*.lib' \) -delete
