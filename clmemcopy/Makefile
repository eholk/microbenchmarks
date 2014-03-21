ifeq ($(shell uname), Darwin)
CFLAGS=-framework OpenCL
else
CFLAGS= -I/usr/local/cuda/include -L/usr/lib64/nvidia -lOpenCL -lrt
endif

all: example chunks

example: main.c
	g++ -O2 -o example main.c $(CFLAGS)

chunks: chunks.c
	g++ -O2 -o chunks chunks.c $(CFLAGS)
