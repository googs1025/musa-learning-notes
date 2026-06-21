#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

// unroll2：每个线程一次加载两个元素，先在寄存器里合并，再写 shared。
// 好处是减少 block 数和 shared 归约轮数，通常比 naive 版本更接近带宽上限。
__global__ void reduce_unroll2(const float* in, float* partial, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float v = 0.0f;
    if (i < n) v += in[i];
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    s[tid] = v;
    __syncthreads();

    // shared memory 内仍然用折半归约，方便和 02_reduce_naive 做单变量对比。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

int main() {
    const int N = 1 << 22;
    const int threads = 256;
    const int blocks = (N + threads * 2 - 1) / (threads * 2);
    float *h = (float*)std::malloc(N * sizeof(float));
    float *hp = (float*)std::malloc(blocks * sizeof(float));
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d = nullptr, *p = nullptr;
    MUSA_CHECK(musaMalloc(&d, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&p, blocks * sizeof(float)));
    MUSA_CHECK(musaMemcpy(d, h, N * sizeof(float), musaMemcpyHostToDevice));
    GpuTimer t; t.start();
    reduce_unroll2<<<blocks, threads, threads * sizeof(float)>>>(d, p, N);
    t.stop(); MUSA_CHECK_KERNEL();
    MUSA_CHECK(musaMemcpy(hp, p, blocks * sizeof(float), musaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += hp[i];
    std::printf("sum=%.0f expected=%d kernel=%.3f ms partial_blocks=%d\n", sum, N, t.elapsed_ms(), blocks);
    MUSA_CHECK(musaFree(d)); MUSA_CHECK(musaFree(p)); std::free(h); std::free(hp);
}
