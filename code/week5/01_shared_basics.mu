#include "musa_common.h"
#include <cstdio>
__global__ void static_shared(float* out){__shared__ float s[256]; int t=threadIdx.x; s[t]=float(t); __syncthreads(); out[t]=s[255-t];}
__global__ void dynamic_shared(float* out){extern __shared__ float s[]; int t=threadIdx.x; s[t]=float(t*2); __syncthreads(); out[t]+=s[t];}
int main(){float* d=nullptr; MUSA_CHECK(musaMalloc(&d,256*sizeof(float))); static_shared<<<1,256>>>(d); MUSA_CHECK_KERNEL(); dynamic_shared<<<1,256,256*sizeof(float)>>>(d); MUSA_CHECK_KERNEL(); std::printf("shared basics done\n"); MUSA_CHECK(musaFree(d));}
