NVCC_OPTS = -arch sm_21 -I$(HOME)/local/include -L$(HOME)/local/lib -lpapi

all: nbody nbody-soa

nbody: nbody.cu Makefile
	nvcc nbody.cu -o nbody $(NVCC_OPTS)

nbody-soa: nbody-soa.cu Makefile
	nvcc nbody-soa.cu -o nbody-soa $(NVCC_OPTS)
