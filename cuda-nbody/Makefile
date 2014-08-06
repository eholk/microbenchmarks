NVCC_OPTS = -arch sm_13 -I$(HOME)/local/include -L$(HOME)/local/lib -lpapi -O3

all: nbody nbody-soa nbody-vector nbody-strided

nbody: nbody.cu Makefile
	nvcc nbody.cu -o nbody $(NVCC_OPTS)

nbody-soa: nbody-soa.cu Makefile
	nvcc nbody-soa.cu -o nbody-soa $(NVCC_OPTS)

nbody-vector: nbody-vector.cu Makefile
	nvcc nbody-vector.cu -o nbody-vector $(NVCC_OPTS)

nbody-strided: nbody-strided.cu Makefile
	nvcc nbody-strided.cu -o nbody-strided $(NVCC_OPTS)
