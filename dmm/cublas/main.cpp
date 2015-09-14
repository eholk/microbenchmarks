#include <iostream>
#include <cassert>
#include <iomanip>

#include <string.h>
#include <malloc.h>
#include <papi.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

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

// Generate a random vector
float *generate_vector(int N) {
    float *v = (float *)memalign(32,
                                   sizeof(float) * N * N);

    for(int i = 0; i < N * N; ++i) {
        v[i] = 1.0;
    }

    return v;
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

int main() {
	cublasCreate(&handle);

	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

	const int K = 10;
	
    for(int i = 1; i <= 134; i+=2) {
        const int N = 100 * i;

        float *A = generate_vector(N);
        float *B = generate_vector(N);

        long long start = PAPI_get_real_usec();

        for(int i = 0; i < K; ++i)
            cublas_dmm(N, A, B);

        long long stop = PAPI_get_real_usec();

        cout << N << "\t" << double(stop - start) / K << endl;

        free(A);
        free(B);
    }

    cublasDestroy(handle);
}
