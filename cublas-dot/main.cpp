#include <iostream>
#include <cassert>
#include <iomanip>

#include <string.h>
#include <malloc.h>
#include <papi.h>

#include <cublas.h>

using namespace std;


// Generate a random vector
float *generate_vector(int N) {
    float *v = (float *)memalign(32,
                                   sizeof(float) * N);

    for(int i = 0; i < N; ++i) {
        v[i] = 1.0;
    }

    return v;
}

float cublas_dot(int N, float *A, float *B) {
    float *Ad;
    float *Bd;
    cublasAlloc(N, sizeof(float), (void **)&Ad);
    cublasAlloc(N, sizeof(float), (void **)&Bd);

    cublasSetVector(N, sizeof(float), A, 1, Ad, 1);
    cublasSetVector(N, sizeof(float), B, 1, Bd, 1);

    float dot = cublasSdot(N, Ad, 1, Bd, 1);
    assert(cublasGetError() == CUBLAS_STATUS_SUCCESS);

    cublasFree(Ad);
    cublasFree(Bd);

    return dot;
}

int main() {
    cublasInit();

    cout << "name: cublas-dotprod" << endl
         << "results:" << endl;
    
    for(int i = 1; i <= 134; i+=2) {
        const int N = 1000000 * i;

        float *A = generate_vector(N);
        float *B = generate_vector(N);

        cout << "results:" << endl
             << "- vector_size: " << N << endl
             << "  raw_data:" << endl
             << "    total_time:" << endl;
        

        for(int i = 0; i < 10; ++i) {
	        long long start = PAPI_get_real_usec();
            cublas_dot(N, A, B);
            long long stop = PAPI_get_real_usec();
            cout << "    - " << double(stop - start) / 1e9 << endl;
        }

        free(A);
        free(B);
    }

    cublasShutdown();
}
