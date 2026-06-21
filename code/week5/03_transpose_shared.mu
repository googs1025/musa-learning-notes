#include "musa_common.h"
#include <cstdio>
constexpr int TILE=32;
__global__ void transpose_padded(const float* in,float* out,int w,int h){__shared__ float tile[TILE][TILE+1]; int x=blockIdx.x*TILE+threadIdx.x; int y=blockIdx.y*TILE+threadIdx.y; if(x<w&&y<h) tile[threadIdx.y][threadIdx.x]=in[y*w+x]; __syncthreads(); x=blockIdx.y*TILE+threadIdx.x; y=blockIdx.x*TILE+threadIdx.y; if(x<h&&y<w) out[y*h+x]=tile[threadIdx.x][threadIdx.y];}
int main(){const int W=2048,H=2048; float *in=nullptr,*out=nullptr; MUSA_CHECK(musaMalloc(&in,W*H*sizeof(float))); MUSA_CHECK(musaMalloc(&out,W*H*sizeof(float))); dim3 b(TILE,TILE),g((W+TILE-1)/TILE,(H+TILE-1)/TILE); transpose_padded<<<g,b>>>(in,out,W,H); MUSA_CHECK_KERNEL(); GpuTimer t;t.start(); for(int r=0;r<20;++r) transpose_padded<<<g,b>>>(in,out,W,H); t.stop(); MUSA_CHECK(musaDeviceSynchronize()); std::printf("transpose shared padded %.3f ms\n",t.elapsed_ms()/20.0f); MUSA_CHECK(musaFree(in)); MUSA_CHECK(musaFree(out));}
