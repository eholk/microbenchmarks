CXX=clang++

INCLUDE=-I../../cppbench/include
LIBS=-lOpenCL

CXXFLAGS=$(INCLUDE) $(LIBS) \
	-std=c++11

opencl-dmm: main.cpp
	$(CXX) $(CXXFLAGS) main.cpp -o opencl-dmm

.phony: clean
clean:
	rm -f opencl-dmm
