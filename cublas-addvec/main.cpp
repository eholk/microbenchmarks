#include <iostream>
#include <cassert>
#include <iomanip>

#include <string.h>
#include <malloc.h>
#include <papi.h>

#include <cublas.h>

using namespace std;

const int NUM_TRIALS = 10;

const int MAX_SIZE = 89000000;
const int STEP = 2000000;

// Generate a random vector
float *generate_vector(int N) {
    float *v = (float *)memalign(32,
                                   sizeof(float) * N);

    for(int i = 0; i < N; ++i) {
        v[i] = 1.0;
    }

    return v;
}

void cublas_addvec(int N, float *A, float *B) {
    float *Ad;
    float *Bd;
    cublasAlloc(N, sizeof(float), (void **)&Ad);
    cublasAlloc(N, sizeof(float), (void **)&Bd);

    cublasSetVector(N, sizeof(float), A, 1, Ad, 1);
    cublasSetVector(N, sizeof(float), B, 1, Bd, 1);

    cublasSaxpy(N, 1.0, Ad, 1, Bd, 1);
    assert(cublasGetError() == CUBLAS_STATUS_SUCCESS);

    cublasFree(Ad);
    cublasFree(Bd);
}

int main() {
    cublasInit();

    for(int N = 1000000; N <= MAX_SIZE; N += STEP) {
        float *A = generate_vector(N);
        float *B = generate_vector(N);

        long long start = PAPI_get_real_usec();

        for(int i = 0; i < NUM_TRIALS; ++i) {
          cublas_addvec(N, A, B);
        }

        long long stop = PAPI_get_real_usec();

        cout << N << "\t" << (stop - start) / NUM_TRIALS << endl;

        free(A);
        free(B);
    }
    
    cublasShutdown();
}
