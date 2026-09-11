// 预计输出：
//   sum=4194304 expected=4194304 kernel=... ms partial_blocks=8192
// 注意：相对 naive reduce，unroll2 通常减少 block 数并改善耗时。

#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

// unroll2：每个线程一次加载两个元素，先在寄存器里合并，再写 shared。
// 好处是减少 block 数和部分调度开销；是否更快仍需在相同计时口径下实测。
//
// 以 blockDim.x = 4、blockIdx.x = 0 为例：
//
//   thread 0: 读取 in[0] 和 in[4]，先算 v0 = in[0] + in[4]
//   thread 1: 读取 in[1] 和 in[5]，先算 v1 = in[1] + in[5]
//   thread 2: 读取 in[2] 和 in[6]，先算 v2 = in[2] + in[6]
//   thread 3: 读取 in[3] 和 in[7]，先算 v3 = in[3] + in[7]
//
//   shared s[]: [v0, v1, v2, v3]
//                  └── 后面继续执行与 02 相同的折半归约 ──┘
//
// 与 02 的区别是：一个 block 覆盖 2 * blockDim.x 个输入元素，
// 所以相同 N 下需要的 block 数约减半；两个加载位置都要独立做边界检查。
__global__ void reduce_unroll2(const float* in, float* partial, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x * 2 + tid;
    float v = 0.0f;
    // 第一个元素位于当前 block 的前半段，第二个元素位于后半段。
    if (i < n) v += in[i];
    if (i + blockDim.x < n) v += in[i + blockDim.x];
    s[tid] = v;
    __syncthreads();

    // 两个 global 元素先在寄存器 v 中合并，写入 shared 后再做 block 内归约。
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

    // 和 02 保持相同的单次 kernel 计时口径，便于只比较“每线程加载 2 个元素”
    // 对归约结构的影响。这里不是重复 50 次的稳定 benchmark，首次 launch
    // 开销和设备状态仍可能影响这个教学示例的单次数字。
    GpuTimer t; t.start();
    reduce_unroll2<<<blocks, threads, threads * sizeof(float)>>>(d, p, N);
    t.stop(); MUSA_CHECK_KERNEL();
    MUSA_CHECK(musaMemcpy(hp, p, blocks * sizeof(float), musaMemcpyDeviceToHost));
    double sum = 0.0;
    for (int i = 0; i < blocks; ++i) sum += hp[i];
    std::printf("sum=%.0f expected=%d kernel=%.3f ms partial_blocks=%d\n", sum, N, t.elapsed_ms(), blocks);
    MUSA_CHECK(musaFree(d)); MUSA_CHECK(musaFree(p)); std::free(h); std::free(hp);
}
