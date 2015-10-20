#pragma once

#include <malloc.h>

#include "cppbench.hpp"

// Generate a random vector
float *generate_vector(int N) {
    float *v = (float *)memalign(32,
                                   sizeof(float) * N * N);

    for(int i = 0; i < N * N; ++i) {
        v[i] = 1.0;
    }

    return v;
}

class DmmBenchmark : public Benchmark
{
protected:
	int N;
	float *A;
	float *B;

public:
	DmmBenchmark(int N) : N(N), A(nullptr), B(nullptr) {}
	
	virtual void setup() {
		A = generate_vector(N);
		B = generate_vector(N);
	}

	virtual void cleanup() {
		delete A; A = nullptr;
		delete B; B = nullptr;
	}
};
