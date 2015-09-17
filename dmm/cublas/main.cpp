#include <iostream>
#include <cassert>
#include <iomanip>

#include <string.h>
#include <malloc.h>
#include <papi.h>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "cppbench.hpp"

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

class CublasDmmBenchmark : public Benchmark
{
	int N;
	float *A;
	float *B;
	
public:
	CublasDmmBenchmark(int N) : N(N), A(nullptr), B(nullptr) {}
	
	virtual void setup() {
		A = generate_vector(N);
		B = generate_vector(N);
	}

	virtual void cleanup() {
		delete A; A = nullptr;
		delete B; B = nullptr;
	}

	virtual void run_iteration() {
		cublas_dmm(N, A, B);
	}
};

int main() {
	cublasCreate(&handle);

	cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    cout << "---" << endl;
    cout << "name: cublas-dmm" << endl;

    cout << "results:" << endl;
    for(int i = 1; i <= 134; i+=2) {
        const int N = 100 * i;

        AdvancedBenchmarkRunner runner;
        CublasDmmBenchmark bench(N);

        runner.run(bench);

        auto width = runner.confidenceWidth();
        auto interval = runner.confidenceInterval();

        cout << "- matrix_size: " << N << endl;

        cout << "  raw_data:" << endl
             << "    total_time:" << endl;
        for(auto i = 0; i < runner.getNumTrials(); ++i) {
            cout << "    - " << runner.getSample(i) << endl;
        }

        cout << "  summary:" << endl
             << "    total_time:" << endl
             << "      sample_size:         " << runner.getNumTrials() << endl
             << "      mean:                " << runner.timePerIteration() << endl
             << "      std_dev:             " << runner.getStdDev() << endl
             << "      confidence_width:    " << width << endl
             << "      confidence_interval: [" << get<0>(interval) << ", " << get<1>(interval) << "]"
             << endl;
    }

    cublasDestroy(handle);
    cout << "..." << endl;
}