#include <iostream>
#include "../common/common.h"

using namespace std;

const int N = 33554432;
const int TRIALS = 5;

void time_scale(int scale, int N, double *xs, double *rs) {
	double start = time_s();

	for(int k = 0; k < TRIALS; ++k) {
		for(int i = 0; i < N; ++i) {
			// use the Maclaurin series for 1/(1-x)
			double x = xs[i];
			double t = 1.0;
			for(int j = 0; j < scale; ++j) {
				// each scale gives one add and one mul, although it
				// doesn't look like these can be fused.
				t += x;
				x *= xs[i];
			}
			rs[i] = t;
		}
	}
	
	double end = time_s();
	cout << scale << "\t" << (end - start) / TRIALS << endl;
}

int main() {
	double *xs = generate_vector(N);
	double *rs = NULL;
	posix_memalign((void**)&rs, 32, sizeof(double) * N);

	for(int scale = 0; scale < 256; ++scale) {
		time_scale(scale, N, xs, rs);
	}
	
	free(xs);
	free(rs);
}
