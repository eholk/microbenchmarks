/* -*- c++ -*- */

// This is the first version. We'll use array of structs.

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

int main() {
	return 0;
}
