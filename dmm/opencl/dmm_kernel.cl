/* -*- c -*- */


__kernel
void dmm(int N, __global float *A, __global float *B, __global float *C) {
	int i = get_global_id(0);
	int j = get_global_id(1);

#define ref(A, i, j) ((A)[(i) * N + (j)])
	
	float acc = 0;
	for(int k = 0; k < N; ++k) {
		acc += ref(A, i, k) * ref(B, k, j);
	}

	ref(C, i, j) = acc;
}
