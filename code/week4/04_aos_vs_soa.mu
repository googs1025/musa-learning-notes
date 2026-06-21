#include "musa_common.h"
#include <cstdio>
struct Particle{float x,y,z,w;};
__global__ void aos(const Particle* p,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=p[i].x+p[i].y;}
__global__ void soa(const float* x,const float* y,float* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) out[i]=x[i]+y[i];}
int main(){const int N=1<<24; Particle* p=nullptr; float *x=nullptr,*y=nullptr,*out=nullptr; MUSA_CHECK(musaMalloc(&p,N*sizeof(Particle))); MUSA_CHECK(musaMalloc(&x,N*sizeof(float))); MUSA_CHECK(musaMalloc(&y,N*sizeof(float))); MUSA_CHECK(musaMalloc(&out,N*sizeof(float))); dim3 b(256),g((N+255)/256); GpuTimer t; aos<<<g,b>>>(p,out,N); MUSA_CHECK_KERNEL(); t.start(); for(int r=0;r<30;++r)aos<<<g,b>>>(p,out,N); t.stop(); float ta=t.elapsed_ms()/30.0f; soa<<<g,b>>>(x,y,out,N); MUSA_CHECK_KERNEL(); t.start(); for(int r=0;r<30;++r)soa<<<g,b>>>(x,y,out,N); t.stop(); float ts=t.elapsed_ms()/30.0f; std::printf("AoS=%.3f ms SoA=%.3f ms speedup=%.2fx\n",ta,ts,ta/ts); MUSA_CHECK(musaFree(p)); MUSA_CHECK(musaFree(x)); MUSA_CHECK(musaFree(y)); MUSA_CHECK(musaFree(out));}
