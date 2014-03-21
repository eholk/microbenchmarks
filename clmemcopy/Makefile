ifeq ($(shell uname), Darwin)
CFLAGS=-framework OpenCL
else
CFLAGS= -I/usr/local/cuda/include -L/usr/lib64/nvidia -lOpenCL -lrt
endif

example: main.c
	g++ -o example main.c $(CFLAGS)
