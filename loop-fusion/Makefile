.PHONY: all
all: fusion-g++.s fusion-clang++.s fusion-g++ fusion-clang++ fusion-clang++.ll

.PHONY: clean
clean:
	rm -f *.s fusion-g++ fusion-clang++ *.ll *.bc

.PHONY: clang
clang: fusion-clang++.s fusion-clang++.ll fusion-clang++

CXXFLAGS = -O4

fusion-g++.s : fusion.cpp
	g++ -S $(CXXFLAGS) fusion.cpp -o fusion-g++.s

fusion-clang++.s : fusion.cpp
	clang++ -S $(CXXFLAGS) fusion.cpp -o fusion-clang++.s

fusion-clang++.ll : fusion.cpp
	clang++ -S -emit-llvm $(CXXFLAGS) fusion.cpp -o fusion-clang++.ll
#	llvm-dis fusion-clang++.bc

fusion-g++ : fusion.cpp
	g++ $(CXXFLAGS) fusion.cpp -o fusion-g++

fusion-clang++ : fusion.cpp
	llvm-as llfuncs.ll > llfuncs.bc
	clang++ -emit-llvm-bc $(CXXFLAGS) -c fusion.cpp -o fusion.bc
	llvm-link llfuncs.bc fusion.bc -o all.bc
	opt -O3 all.bc -o optimized.bc
	llc optimized.bc -o fusion-clang++.s
	clang++ fusion-clang++.s -o fusion-clang++
