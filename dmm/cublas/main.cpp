#include <iostream>
#include <cassert>
#include <iomanip>

#include <string.h>
#include <papi.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "../common.hpp"

using namespace std;

cublasHandle_t handle;

#define CUBLAS_CHECK(e) {	  \
		auto st = e; \
		cublas_report(st); \
		assert(st == CUBLAS_STATUS_SUCCESS); \
	}

void cublas_report(cublasStatus_t status) {
#define check(s) if(status == s) cerr << #s << endl;
	check(CUBLAS_STATUS_NOT_INITIALIZED);
	check(CUBLAS_STATUS_ALLOC_FAILED);
	check(CUBLAS_STATUS_INVALID_VALUE);
	check(CUBLAS_STATUS_ARCH_MISMATCH);
	check(CUBLAS_STATUS_MAPPING_ERROR);
	check(CUBLAS_STATUS_EXECUTION_FAILED);
	check(CUBLAS_STATUS_INTERNAL_ERROR);
	check(CUBLAS_STATUS_NOT_SUPPORTED);
	check(CUBLAS_STATUS_LICENSE_ERROR);
#undef check
}

void cublas_dmm(int N, float *A, float *B) {
    float *Ad;
    float *Bd;
    float *Cd;
    float alpha = 1;
    float beta = 0;
    cudaMalloc((void **)&Ad, N * N * sizeof(float));
    cudaMalloc((void **)&Bd, N * N * sizeof(float));
    cudaMalloc((void **)&Cd, N * N * sizeof(float));

    CUBLAS_CHECK(cublasSetMatrix(N, N, sizeof(float), A, N, Ad, N));
    CUBLAS_CHECK(cublasSetMatrix(N, N, sizeof(float), B, N, Bd, N));

    auto status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                              N, N, N, &alpha,
                              Ad, N, Bd, N,
                              &beta,
                              Cd, N);
    cublas_report(status);
    assert(status == CUBLAS_STATUS_SUCCESS);

    cudaFree(Ad);
    cudaFree(Bd);
    cudaFree(Cd);
}

class CublasDmmBenchmark : public DmmBenchmark
{	
public:
	CublasDmmBenchmark(int N, Params &params) : DmmBenchmark(N) {}
	
	virtual void run_iteration() {
		cublas_dmm(N, A, B);
	}
};

int main() {
	cublasCreate(&handle);

	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    cout << "name: cublas-dmm" << endl;

    run_benchmark<CublasDmmBenchmark>();
    
    cublasDestroy(handle);
}
