// 预计输出：
//   sum=4194304 expected=4194304 kernel=... ms partial_blocks=16384
// 注意：kernel 时间随设备变化；sum 必须等于 expected。

#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

// 每个 block 负责输入数组的一段数据，先在 shared memory 内归约出一个 partial sum。
// 最终跨 block 的求和先放回 host 做，保持第一版 reduce 足够简单。
//
// 以 blockDim.x = 8、blockIdx.x = 0 为例：
//
//   global input:  in[0] in[1] in[2] in[3] in[4] in[5] in[6] in[7]
//                    │     │     │     │     │     │     │     │
//   threadIdx.x:      0     1     2     3     4     5     6     7
//                    └──────────────┬──────────────────────────┘
//                                   load
//                    ┌─────────────▼───────────────────────────┐
//   shared s[]:     [in0,  in1,  in2,  in3,  in4,  in5,  in6,  in7]
//
//   reduce tree:
//       stride=4: s[0]+=s[4], s[1]+=s[5], s[2]+=s[6], s[3]+=s[7]
//       stride=2: s[0]+=s[2], s[1]+=s[3]
//       stride=1: s[0]+=s[1]
//
//   最后只有 thread 0 拥有这个 block 的总和，并写入 partial[blockIdx.x]。
__global__ void reduce_naive(const float* in, float* partial, int n) {
    extern __shared__ float s[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;

    // 第一步：一个 thread 处理一个 global 元素，并把它搬到 shared memory。
    // 最后一个 block 可能覆盖到 n 之外；越界线程填 0，便于后面统一相加。
    s[tid] = (i < n) ? in[i] : 0.0f;

    // 等待整个 block 的所有线程完成 load。
    // 如果没有这个同步，某个线程可能在其他线程写完 s[] 之前就开始读取。
    __syncthreads();

    // 第二步：折半归约，每一轮把“后半段”加到“前半段”。
    //
    //   stride = 4: thread 0..3 读取 s[4..7]
    //   stride = 2: thread 0..1 读取上一轮的 s[2..3]
    //   stride = 1: thread 0   读取上一轮的 s[1]
    //
    // 因此每轮活跃线程数减半；后几轮只有少数线程工作，可能产生分支发散。
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        // tid >= stride 的线程本轮不写，只等待下一次同步。
        if (tid < stride) s[tid] += s[tid + stride];

        // 当前轮的写入结果会被下一轮读取，所以每轮都必须同步。
        __syncthreads();
    }

    // 第三步：每个 block 只输出一个 partial sum。
    // 不同 block 不能用 __syncthreads() 直接同步，因此跨 block 的最终求和
    // 留给 host，或由后续的第二个 reduction kernel 完成。
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

    // 这里只测一次 kernel：GpuTimer 的 event start/stop 都在默认 stream，
    // 所以测到的是这次 reduce kernel 在 GPU 时间线上的执行时间。
    // t.stop() 后的 MUSA_CHECK_KERNEL 会等待 kernel 完成，再安全地把 partial
    // 拷回 host；它不是另一个归约步骤。
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
