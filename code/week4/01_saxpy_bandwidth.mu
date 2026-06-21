#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

__global__ void saxpy(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
int main(){
    const int N = 1 << 24; size_t bytes = N * sizeof(float);
    float *x=nullptr,*y=nullptr; MUSA_CHECK(musaMalloc(&x,bytes)); MUSA_CHECK(musaMalloc(&y,bytes));
    MUSA_CHECK(musaMemset(x,0,bytes)); MUSA_CHECK(musaMemset(y,0,bytes));
    dim3 block(256), grid((N+255)/256); saxpy<<<grid,block>>>(2.0f,x,y,N); MUSA_CHECK_KERNEL();
    GpuTimer t; t.start(); for(int i=0;i<50;++i) saxpy<<<grid,block>>>(2.0f,x,y,N); t.stop(); MUSA_CHECK(musaDeviceSynchronize());
    float ms=t.elapsed_ms()/50.0f; double gb=(3.0*bytes)/1e9; std::printf("SAXPY %.3f ms %.2f GB/s\n",ms,gb/(ms/1000.0));
    MUSA_CHECK(musaFree(x)); MUSA_CHECK(musaFree(y));
}
