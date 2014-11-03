#pragma once

#include <stdlib.h>
#include <inttypes.h>
#include <time.h>

#ifdef __APPLE__
#include <mach/mach_time.h>
#endif

uint64_t nanotime() {
#ifdef __APPLE__
    uint64_t time = mach_absolute_time();
    mach_timebase_info_data_t info = {0, 0};
    if (info.denom == 0) {
        mach_timebase_info(&info);
    }
    uint64_t time_nano = time * (info.numer / info.denom);
    return time_nano;  
#else
    uint64_t ns_per_s = 1000000000LL;
    timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ((uint64_t)ts.tv_sec * (uint64_t)ns_per_s + (uint64_t)ts.tv_nsec);
#endif    
}

double time_s() {
	return double(nanotime() / 1000) / 1e6;
}

// Generate a random vector
double *generate_vector(int N) {
    double *v = NULL;
	posix_memalign((void **)&v, 32, sizeof(double) * N);

    for(int i = 0; i < N; ++i) {
        v[i] = drand48();
    }

    return v;
}
