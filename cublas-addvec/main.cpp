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
double *generate_vector(int N) {
    double *v = (double *)memalign(32,
                                   sizeof(double) * N);

    for(int i = 0; i < N; ++i) {
        v[i] = 1.0;
    }

    return v;
}

void cublas_addvec(int N, double *A, double *B) {
    double *Ad;
    double *Bd;
    cublasAlloc(N, sizeof(double), (void **)&Ad);
    cublasAlloc(N, sizeof(double), (void **)&Bd);

    cublasSetVector(N, sizeof(double), A, 1, Ad, 1);
    cublasSetVector(N, sizeof(double), B, 1, Bd, 1);

    cublasDaxpy(N, 1.0, Ad, 1, Bd, 1);
    assert(cublasGetError() == CUBLAS_STATUS_SUCCESS);

    cublasFree(Ad);
    cublasFree(Bd);
}

int main() {
    cublasInit();

    for(int N = 1000000; N <= MAX_SIZE; N += STEP) {
        double *A = generate_vector(N);
        double *B = generate_vector(N);

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
