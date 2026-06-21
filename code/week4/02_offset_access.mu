#include "musa_common.h"
#include <cstdio>

__global__ void offset_copy(const float* in, float* out, int n, int offset){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=in[i+offset];}
float run(const float* in,float* out,int n,int off){dim3 b(256),g((n+255)/256); offset_copy<<<g,b>>>(in,out,n,off); MUSA_CHECK_KERNEL(); GpuTimer t;t.start(); for(int r=0;r<50;++r) offset_copy<<<g,b>>>(in,out,n,off); t.stop(); MUSA_CHECK(musaDeviceSynchronize()); return t.elapsed_ms()/50.0f;}
int main(){const int N=1<<24; float *in=nullptr,*out=nullptr; MUSA_CHECK(musaMalloc(&in,(N+64)*sizeof(float))); MUSA_CHECK(musaMalloc(&out,N*sizeof(float))); for(int off: {0,1,2,4,8,16,31}) std::printf("offset=%2d time=%.3f ms\n",off,run(in,out,N,off)); MUSA_CHECK(musaFree(in)); MUSA_CHECK(musaFree(out));}
