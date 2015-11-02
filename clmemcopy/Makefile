ifeq ($(shell uname), Darwin)
CFLAGS=-framework OpenCL
else
CFLAGS= -I/usr/local/cuda/include -L/usr/lib64/nvidia -lOpenCL -lrt
endif

all: example chunks

plots: memcopy.csv memcopy-chunked.csv
	python mkplots.py

data: memcopy.csv memcopy-chunked.csv

example: main.c
	g++ -O2 -o example main.c $(CFLAGS)

chunks: chunks.c
	g++ -O2 -o chunks chunks.c $(CFLAGS)

memcopy.csv: example
	./example > memcopy.csv

memcopy-chunked.csv: chunks
	./chunks > memcopy-chunked.csv

.phony: clean
clean:
	rm -f example chunks
