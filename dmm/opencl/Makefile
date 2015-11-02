CXX=clang++

INCLUDE=-I../../cppbench/include
LIBS=-lOpenCL

CXXFLAGS=$(INCLUDE) $(LIBS) \
	-std=c++11

opencl-dmm: main.cpp dmm_kernel.h ../common.hpp
	$(CXX) $(CXXFLAGS) main.cpp -o opencl-dmm

dmm_kernel.h: dmm_kernel.cl
	xxd -i dmm_kernel.cl > $@

.phony: clean
clean:
	rm -f opencl-dmm
