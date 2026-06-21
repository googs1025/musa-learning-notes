#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

__inline__ __device__ float warp_reduce_sum(float v) {
    // MUSA warp size is often 128. Confirm shuffle mask/signature with the local SDK.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    return v;
}

__global__ void reduce_shfl(const float* in, float* partial, int n) {
    extern __shared__ float warp_sums[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    float v = (i < n) ? in[i] : 0.0f;
    v = warp_reduce_sum(v);
    int lane = tid % warpSize;
    int warp = tid / warpSize;
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (tid < (blockDim.x + warpSize - 1) / warpSize) ? warp_sums[lane] : 0.0f;
        v = warp_reduce_sum(v);
        if (tid == 0) partial[blockIdx.x] = v;
    }
}

int main() {
    const int N = 1 << 20;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    float *h = (float*)std::malloc(N * sizeof(float));
    float *hp = (float*)std::malloc(blocks * sizeof(float));
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d = nullptr, *p = nullptr;
    MUSA_CHECK(musaMalloc(&d, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&p, blocks * sizeof(float)));
    MUSA_CHECK(musaMemcpy(d, h, N * sizeof(float), musaMemcpyHostToDevice));
    int shared = ((threads + 127) / 128) * sizeof(float);
    reduce_shfl<<<blocks, threads, shared>>>(d, p, N);
    MUSA_CHECK_KERNEL();
    MUSA_CHECK(musaMemcpy(hp, p, blocks * sizeof(float), musaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += hp[i];
    std::printf("sum=%.0f expected=%d warpSize=%d\n", sum, N, warpSize);
    MUSA_CHECK(musaFree(d)); MUSA_CHECK(musaFree(p)); std::free(h); std::free(hp);
}
