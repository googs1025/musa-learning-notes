#include "musa_common.h"
#include <cstdio>
#include <cstdlib>
__constant__ float c_w[5];
__global__ void stencil5(const float* in,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=2&&i<n-2){float v=0; for(int k=-2;k<=2;++k)v+=in[i+k]*c_w[k+2]; out[i]=v;}}
int main(){const int N=1<<20; float w[5]={0.0625f,0.25f,0.375f,0.25f,0.0625f}; float *in=nullptr,*out=nullptr; MUSA_CHECK(musaMalloc(&in,N*sizeof(float))); MUSA_CHECK(musaMalloc(&out,N*sizeof(float))); MUSA_CHECK(musaMemcpyToSymbol(c_w,w,sizeof(w))); dim3 b(256),g((N+255)/256); stencil5<<<g,b>>>(in,out,N); MUSA_CHECK_KERNEL(); std::printf("constant stencil done\n"); MUSA_CHECK(musaFree(in)); MUSA_CHECK(musaFree(out));}
