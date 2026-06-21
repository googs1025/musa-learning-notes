#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

// 每个 block 先在 shared memory 内归约出一个 partial sum。
// 最终跨 block 的求和先放回 host 做，保持第一版 reduce 足够简单。
__global__ void reduce_naive(const float* in, float* partial, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    s[tid] = (i < n) ? in[i] : 0.0f;
    __syncthreads();

    // 折半归约：每轮活跃线程减半。这个版本容易理解，但后几轮会有分支发散。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = s[0];
}

int main() {
    const int N = 1 << 22;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    float *h = (float*)std::malloc(N * sizeof(float));
    float *hp = (float*)std::malloc(blocks * sizeof(float));
    for (int i = 0; i < N; ++i) h[i] = 1.0f;
    float *d = nullptr, *p = nullptr;
    MUSA_CHECK(musaMalloc(&d, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&p, blocks * sizeof(float)));
    MUSA_CHECK(musaMemcpy(d, h, N * sizeof(float), musaMemcpyHostToDevice));
    GpuTimer t; t.start();
    reduce_naive<<<blocks, threads, threads * sizeof(float)>>>(d, p, N);
    t.stop(); MUSA_CHECK_KERNEL();

    // 教学版：partial 拷回 host 后串行累加。Week3 后续可改成二次 kernel 归约。
    MUSA_CHECK(musaMemcpy(hp, p, blocks * sizeof(float), musaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += hp[i];
    std::printf("sum=%.0f expected=%d kernel=%.3f ms partial_blocks=%d\n", sum, N, t.elapsed_ms(), blocks);
    MUSA_CHECK(musaFree(d)); MUSA_CHECK(musaFree(p)); std::free(h); std::free(hp);
}
