/* -*- c++ -*- */

// This is the first version. We'll use array of structs.

#include <papi.h>

#include <iostream>
using namespace std;

float __device__ mag(float3 p) {
	return sqrt(p.x * p.x + p.y * p.y + p.z * p.z);
}

float3 __device__ operator-(float3 lhs, float3 rhs) {
	float3 result;
	result.x = lhs.x - rhs.x;
	result.y = lhs.y - rhs.y;
	result.z = lhs.z - rhs.z;
	return result;
}

float3 __device__ operator+(float3 lhs, float3 rhs) {
	float3 result;
	result.x = lhs.x + rhs.x;
	result.y = lhs.y + rhs.y;
	result.z = lhs.z + rhs.z;
	return result;
}

float3 __device__ operator/(float3 lhs, float rhs) {
	float3 result;
	result.x = lhs.x / rhs;
	result.y = lhs.y / rhs;
	result.z = lhs.z / rhs;
	return result;
}

__global__ void nbody(int N, float3 *bodies, float3 *result) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= N) return;

	float3 total = {0};
	float3 base = bodies[i];

	for(int j = 0; j < N; ++j) {
		float3 diff = bodies[j] - base;
		float d = mag(diff);
		
		if(d > 0) {
			total = total + (diff / (d * d * d));
		}
	}
	result[i] = total;
}


float3 *mkbodies(int N) {
	float3 *bodies = new float3[N];
	for(int i = 0; i < N; ++i) {
		bodies[i].x = bodies[i].y = bodies[i].z = i;
	}
	return bodies;
}

void print_bodies(float3 *bodies, int N) {
	for(int i = 0; i < N; ++i) {
		cout << bodies[i].x << "\t" << bodies[i].y << "\t" << bodies[i].z << endl;
	}
}

int main() {
	const int N = 65000;
	const int SIZE = N * sizeof(float3);

	cout << "generating bodies..."; cout.flush();
	float3 *bodies = mkbodies(N);
	float3 *forces = new float3[N];
	cout << "done." << endl;

	float3 *dbodies = NULL;
	float3 *dforces = NULL;

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
