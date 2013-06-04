#include <iostream>
#include <math.h>
#include <stdlib.h>

using namespace std;

double baseline(int N, double *data)
{
    double result = 0;

    for(int i = 0; i < N; ++i) {
        result += sin(data[i]);
    }

    for(int i = 0; i < N; ++i) {
        result += cos(data[i]);
    }

    return result;
}

double manually_fused(int N, double *data)
{
    double result = 0;

    for(int i = 0; i < N; ++i) {
        result += sin(data[i]);
        result += cos(data[i]);
    }

    return result;
}

double two_vars(int N, double *data)
{
    double result_s = 0;
    double result_c = 0;

    for(int i = 0; i < N; ++i) {
        result_s += sin(data[i]);
    }

    for(int i = 0; i < N; ++i) {
        result_c += cos(data[i]);
    }

    return result_s + result_c;
}

#define asm_sin(x, t) asm("fsin" : "=t" (t) : "0" (x))
#define asm_cos(x, t) asm("fcos" : "=t" (t) : "0" (x))

double asm_baseline(int N, double *data)
{
    double result = 0;

    for(int i = 0; i < N; ++i) {
        double temp;
        asm_sin(data[i], temp);
        result += temp;
    }

    for(int i = 0; i < N; ++i) {
        double temp;
        asm_cos(data[i], temp);
        result += temp;
    }

    return result;
}

double asm_manually_fused(int N, double *data)
{
    double result = 0;

    for(int i = 0; i < N; ++i) {
        double temp;
        asm_sin(data[i], temp);
        result += temp;
        asm_cos(data[i], temp);
        result += temp;
    }

    return result;
}

double asm_two_vars(int N, double *data)
{
    double result_s = 0;
    double result_c = 0;

    for(int i = 0; i < N; ++i) {
        double temp;
        asm_sin(data[i], temp);
        result_s += temp;
    }

    for(int i = 0; i < N; ++i) {
        double temp;
        asm_cos(data[i], temp);
        result_c += temp;
    }

    return result_s + result_c;
}



// Roughly 32 million doubles, so 256 MB
const int COUNT = 32 << 20;

int main()
{
    double *data = new double[COUNT];
    for(int i = 0; i < COUNT; ++i) {
        data[i] = drand48();
    }

    cout << baseline(COUNT, data) << endl;
    cout << manually_fused(COUNT, data) << endl;
    cout << two_vars(COUNT, data) << endl;

    cout << asm_baseline(COUNT, data) << endl;
    cout << asm_manually_fused(COUNT, data) << endl;
    cout << asm_two_vars(COUNT, data) << endl;

    return 0;
}
