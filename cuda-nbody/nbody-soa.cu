/* -*- c++ -*- */

// This version uses struct-of-arrays.

#include <papi.h>

#include <iostream>
using namespace std;
float __device__ mag(float x, float y, float z) {
	return sqrt(x * x + y * y + z * z);
}

__global__ void nbody(int N,
					  float *bx, float *by, float *bz,
					  float *rx, float *ry, float *rz) {
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	
	if(i >= N) return;

	float tx = 0, ty = 0, tz = 0;
	float x = bx[i], y = by[i], z = bz[i];
	
	for(int j = 0; j < N; ++j) {
		float dx = bx[j] - x;
		float dy = by[j] - y;
		float dz = bz[j] - z;

		float d = mag(dx, dy, dz);
		
		if(d > 0) {
			tx += dx / (d * d * d);
			ty += dy / (d * d * d);
			tz += dz / (d * d * d);
		}
	}
	rx[i] = tx;
	ry[i] = ty;
	rz[i] = tz;
}

void mkbodies(int N, float *x, float *y, float * z) {
	for(int i = 0; i < N; ++i) {
		x[i] = y[i] = z[i] = i;
	}
}

void print_bodies(float *x, float *y, float *z, int N) {
	for(int i = 0; i < N; ++i) {
		cout << x[i] << "\t" << y[i] << "\t" << z[i] << endl;
	}
}

int main() {
	const int N = 65000;
	const int SIZE = N * sizeof(float);

	cout << "generating bodies..."; cout.flush();
	float *bx = new float[N];
	float *by = new float[N];
	float *bz = new float[N];
	float *fx = new float[N];
	float *fy = new float[N];
	float *fz = new float[N];
	mkbodies(N, bx, by, bz);
	cout << "done." << endl;

	float *dbx = NULL;
	float *dby = NULL;
	float *dbz = NULL;

	float *dfx = NULL;
	float *dfy = NULL;
	float *dfz = NULL;

	cout << "allocating device memory..."; cout.flush();
	cudaMalloc(&dbx, SIZE);
	cudaMalloc(&dfx, SIZE);
	cudaMalloc(&dby, SIZE);
	cudaMalloc(&dfy, SIZE);
	cudaMalloc(&dbz, SIZE);
	cudaMalloc(&dfz, SIZE);
	cout << "done." << endl;

	long long start_mem = PAPI_get_real_usec();
	cudaMemcpy(dbx, bx, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dby, by, SIZE, cudaMemcpyHostToDevice);
	cudaMemcpy(dbz, bz, SIZE, cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	long long start_compute = PAPI_get_real_usec();
	const int BLOCK_SIZE = 64;
	nbody<<<(N + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(N, dbx, dby, dbz, dfx, dfy, dfz);
	cudaDeviceSynchronize();
	long long stop_compute = PAPI_get_real_usec();
	
	cudaMemcpy(fx, dfx, SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(fy, dfy, SIZE, cudaMemcpyDeviceToHost);
	cudaMemcpy(fz, dfz, SIZE, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	long long stop_mem = PAPI_get_real_usec();

	cout << "First 10 bodies:" << endl;
	print_bodies(bx, by, bz, 10);
	cout << endl << "First 10 forces:" << endl;
	print_bodies(fx, fy, fz, 10);

	cout << endl;

	cout << "Time (total sec):   " << double(stop_mem - start_mem) / 1e6 << endl;
	cout << "Time (compute sec): " << double(stop_compute - start_compute) / 1e6 << endl;

	cudaFree(dbx);
	cudaFree(dfx);
	cudaFree(dby);
	cudaFree(dfy);
	cudaFree(dbz);
	cudaFree(dfz);

	delete [] bx;
	delete [] fx;
	delete [] by;
	delete [] fy;
	delete [] bz;
	delete [] fz;

	return 0;
}
