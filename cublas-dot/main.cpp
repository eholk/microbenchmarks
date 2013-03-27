#include <iostream>
#include <cassert>
#include <iomanip>

#include <string.h>
#include <malloc.h>
#include <papi.h>

#include <cublas.h>

using namespace std;


// Generate a random vector
double *generate_vector(int N) {
    double *v = (double *)memalign(32,
                                   sizeof(double) * N);

    for(int i = 0; i < N; ++i) {
        v[i] = 1.0;
    }

    return v;
}

double cublas_dot(int N, double *A, double *B) {
    double *Ad;
    double *Bd;
    cublasAlloc(N, sizeof(double), (void **)&Ad);
    cublasAlloc(N, sizeof(double), (void **)&Bd);

    cublasSetVector(N, sizeof(double), A, 1, Ad, 1);
    cublasSetVector(N, sizeof(double), B, 1, Bd, 1);

    double dot = cublasDdot(N, Ad, 1, Bd, 1);
    assert(cublasGetError() == CUBLAS_STATUS_SUCCESS);

    cublasFree(Ad);
    cublasFree(Bd);

    return dot;
}

int main() {
    cublasInit();

    for(int i = 1; i <= 20; i++) {
        const int N = 1000000 * i;

        double *A = generate_vector(N);
        double *B = generate_vector(N);

        long long start = PAPI_get_real_usec();

        cublas_dot(N, A, B);

        long long stop = PAPI_get_real_usec();

        cout << N << "\t" << stop - start << endl;

        free(A);
        free(B);
    }

    cublasShutdown();
}
