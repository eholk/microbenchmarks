/* -*- c++ -*- */

// This is the first version. We'll use array of structs.

#include <papi.h>

#include <iostream>
using namespace std;

struct point3 {
	int32_t tag; // Matches the tag field in Harlan
	float x;
	float y;
	float z;
};

float __device__ mag(point3 p) {
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

point3 __device__ operator-(point3 lhs, point3 rhs) {
	point3 result;
	result.x = lhs.x - rhs.x;
	result.y = lhs.y - rhs.y;
	result.z = lhs.z - rhs.z;
	return result;
}

point3 __device__ operator+(point3 lhs, point3 rhs) {
	point3 result;
	result.x = lhs.x + rhs.x;
	result.y = lhs.y + rhs.y;
	result.z = lhs.z + rhs.z;
	return result;
}

point3 __device__ operator/(point3 lhs, float rhs) {
	point3 result;
	result.x = lhs.x / rhs;
	result.y = lhs.y / rhs;
	result.z = lhs.z / rhs;
	return result;
}

__global__ void nbody(int N, point3 *bodies, point3 *result) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= N) return;

	point3 total = {0};
	point3 base = bodies[i];

	for(int j = 0; j < N; ++j) {
		point3 diff = bodies[j] - base;
		float d = mag(diff);
		
		if(d > 0) {
			total = total + (diff / (d * d * d));
		}
	}
	result[i] = total;
}


point3 *mkbodies(int N) {
	point3 *bodies = new point3[N];
	for(int i = 0; i < N; ++i) {
		bodies[i].x = bodies[i].y = bodies[i].z = i;
	}
	return bodies;
}

void print_bodies(point3 *bodies, int N) {
	for(int i = 0; i < N; ++i) {
		cout << bodies[i].x << "\t" << bodies[i].y << "\t" << bodies[i].z << endl;
	}
}

int main() {
	const int N = 65000;
	const int SIZE = N * sizeof(point3);

	cout << "generating bodies..."; cout.flush();
	point3 *bodies = mkbodies(N);
	point3 *forces = new point3[N];
	cout << "done." << endl;

	point3 *dbodies = NULL;
	point3 *dforces = NULL;

	cout << "allocating device memory..."; cout.flush();
	cudaMalloc(&dbodies, SIZE);
	cudaMalloc(&dforces, SIZE);
	cout << "done." << endl;

	long long start_mem = PAPI_get_real_usec();
	cudaMemcpy(dbodies, bodies, SIZE, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	long long start_compute = PAPI_get_real_usec();
	const int BLOCK_SIZE = 64;
	nbody<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(N, dbodies, dforces);
	cudaDeviceSynchronize();
	long long stop_compute = PAPI_get_real_usec();
	
	cudaMemcpy(forces, dforces, SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	long long stop_mem = PAPI_get_real_usec();

	cout << "First 10 bodies:" << endl;
	print_bodies(bodies, 10);
	cout << endl << "First 10 forces:" << endl;
	print_bodies(forces, 10);

	cout << endl;

	cout << "Time (total sec):   " << double(stop_mem - start_mem) / 1e6 << endl;
	cout << "Time (compute sec): " << double(stop_compute - start_compute) / 1e6 << endl;

	cudaFree(dbodies);
	cudaFree(dforces);

	delete [] bodies;
	delete [] forces;
	return 0;
}
