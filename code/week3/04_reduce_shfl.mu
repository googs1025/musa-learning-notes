// 预计输出：
//   sum=1048576 expected=1048576 warpSize=128
// 注意：如果当前 SDK 的 shuffle mask / warpSize 语义不同，可能需要调整实现。

#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

__inline__ __device__ float warp_reduce_sum(float v) {
    // MUSA warp size is often 128. Confirm shuffle mask/signature with the local SDK.
    // shuffle 在 warp 内直接交换寄存器值，不需要 shared memory 和 __syncthreads。
    // 归约示意（以 8-lane group 为简化图）：
    //
    //   [a b c d e f g h]
    //        offset=4  → [a+e b+f c+g d+h ...]
    //        offset=2  → [a+e+c+g b+f+d+h ...]
    //        offset=1  → [a+b+c+d+e+f+g+h ...]
    //
    // 实际 offset 从 warpSize/2 开始；MUSA 的 warp/group 宽度必须按 SDK 确认。
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

    // 第一级：每个 warp/group 内用 shuffle 把 v 汇总到 lane 0。
    // shuffle 只在当前 warp/group 内交换寄存器，不负责不同 warp 之间的同步。
    // 每个 warp 先产出一个 sum，再把这些 warp sum 放到 shared 里做第二级归约。
    if (lane == 0) warp_sums[warp] = v;
    __syncthreads();
    if (warp == 0) {
        // 第二级：第一个 warp/group 读取所有 warp 的 partial sum，再归约一次。
        // 不足一个完整 warp 的位置填 0，避免把未写入的 shared 元素算进去。
        //
        //   每个 warp 的结果: [sum0, sum1, sum2, ...]
        //                         │
        //                         └─ 第一个 warp 再做一次 warp_reduce_sum
        //                            → partial[blockIdx.x]
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
