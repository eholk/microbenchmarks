// -*- c -*-

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define get(A, N, i, j) ((A)[((i) * (N)) + (j)])

__kernel void MyAdd(const double __global *A,
                    const double __global *B,
                    double __global *C,
                    unsigned long int N)
{
    unsigned long int i = get_global_id(0);

    for(unsigned long int j = 0; j < N; j++) {
        if (i % 2 == 0)
            get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j);
        else
            get(C, N, i, j) = get(A, N, i, j) - get(B, N, i, j);
    }
}

__kernel void MyAdd_col(const double __global *A,
                        const double __global *B,
                        double __global *C,
                        unsigned long int N)
{
    unsigned long int i = get_global_id(0);

    for(unsigned long int j = 0; j < N; j++) {
        if (j % 2 == 0)
            get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j);
        else
            get(C, N, i, j) = get(A, N, i, j) - get(B, N, i, j);
    }
}

__kernel void MyAdd_2D(const double __global *A,
                       const double __global *B,
                       double __global *C,
                       unsigned long int N)
{
    unsigned long int i = get_global_id(0);
    unsigned long int j = get_global_id(1);

    if (i % 2 == 0)
        get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j);
    else
        get(C, N, i, j) = get(A, N, i, j) - get(B, N, i, j);
}

__kernel void MyAdd_2D_col(const double __global *A,
                           const double __global *B,
                           double __global *C,
                           unsigned long int N)
{
    unsigned long int i = get_global_id(0);
    unsigned long int j = get_global_id(1);

    if (j % 2 == 0)
        get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j);
    else
        get(C, N, i, j) = get(A, N, i, j) - get(B, N, i, j);
}

__kernel void MyAdd_2D_unweave(const double __global *A,
                               const double __global *B,
                               double __global *C,
                               unsigned long int N)
{
    unsigned long int i = get_global_id(0);
    unsigned long int j = get_global_id(1);

    if(i < N / 2) {
        unsigned long int i = 2 * i;
        get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j);
    }
    else {
        unsigned long int i = 2 * (i - N / 2) + 1;
        get(C, N, i, j) = get(A, N, i, j) - get(B, N, i, j);
    }
}

__kernel void MyAdd_2D_unweave_col(const double __global *A,
                                   const double __global *B,
                                   double __global *C,
                                   unsigned long int N)
{
    unsigned long int i = get_global_id(0);
    unsigned long int j = get_global_id(1);

    if(j < N / 2) {
        unsigned long int j = 2 * j;
        get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j);
    }
    else {
        unsigned long int j = 2 * (j - N / 2) + 1;
        get(C, N, i, j) = get(A, N, i, j) - get(B, N, i, j);
    }
}

__kernel void MyAdd_2D_nobranch(const double __global *A,
                       const double __global *B,
                       double __global *C,
                       unsigned long int N)
{
    unsigned long int i = get_global_id(0);
    unsigned long int j = get_global_id(1);

    get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j)*(1 - ((i&1)<<1));
}

__kernel void MyAdd_2D_col_nobranch(const double __global *A,
                       const double __global *B,
                       double __global *C,
                       unsigned long int N)
{
    unsigned long int i = get_global_id(0);
    unsigned long int j = get_global_id(1);

    get(C, N, i, j) = get(A, N, i, j) + get(B, N, i, j)*(1 - ((j&1)<<1));
}

