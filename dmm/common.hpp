#pragma once

#include <malloc.h>

#include "cppbench.hpp"

// Generate a random vector
float *generate_vector(int N) {
	float *v = (float *)memalign(32, sizeof(float) * N * N);

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
	float *C;
	
public:
	struct Params {};
	
	DmmBenchmark(int N) : N(N), A(nullptr), B(nullptr) {}
	
	virtual void setup() {
		A = generate_vector(N);
		B = generate_vector(N);
		C = (float *)memalign(32, sizeof(float) * N * N);
	}

	virtual void cleanup() {
		delete A; A = nullptr;
		delete B; B = nullptr;
		delete C; C = nullptr;
	}
};

template<class Benchmark>
void run_benchmark(typename Benchmark::Params &&params
                   = typename Benchmark::Params()) {
    cout << "results:" << endl;
    for(int N = 1; N <= 634; N+=2) {
        AdvancedBenchmarkRunner runner;
        Benchmark bench(N, params);

        runner.setNumTrials(5);
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
}
