.PHONY: all
all: fusion-g++.s fusion-clang++.s fusion-g++ fusion-clang++ fusion-clang++.ll

.PHONY: clean
clean:
	rm -f *.s fusion-g++ fusion-clang++ *.ll *.bc

.PHONY: clang
clang: fusion-clang++.s fusion-clang++.ll fusion-clang++

CXXFLAGS = -O3

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
	clang++ $(CXXFLAGS) fusion.cpp -o fusion-clang++

