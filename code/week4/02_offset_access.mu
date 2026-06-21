#include "musa_common.h"
#include <cstdio>

// 目标：让每个线程读取 in[i + offset]，观察起始地址偏移对合并访存的影响。
// offset=0 通常最友好；offset 不是 cache line / transaction 对齐时可能拆成更多事务。
__global__ void offset_copy(const float* in, float* out, int n, int offset) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i + offset];
}

float run(const float* in, float* out, int n, int off) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // warmup：第一次 launch 可能包含缓存 / 调度冷启动开销，不计入正式数据。
    offset_copy<<<grid, block>>>(in, out, n, off);
    MUSA_CHECK_KERNEL();

    GpuTimer t;
    t.start();
    for (int r = 0; r < 50; ++r) {
        offset_copy<<<grid, block>>>(in, out, n, off);
    }
    t.stop();
    MUSA_CHECK(musaDeviceSynchronize());
    return t.elapsed_ms() / 50.0f;
}

int main() {
    const int N = 1 << 24;
    float *in = nullptr, *out = nullptr;

    // 多分配 64 个元素，保证最大 offset=31 时 in[i+offset] 不越界。
    MUSA_CHECK(musaMalloc(&in, (N + 64) * sizeof(float)));
    MUSA_CHECK(musaMalloc(&out, N * sizeof(float)));

    for (int off : {0, 1, 2, 4, 8, 16, 31}) {
        std::printf("offset=%2d time=%.3f ms\n", off, run(in, out, N, off));
    }

    MUSA_CHECK(musaFree(in));
    MUSA_CHECK(musaFree(out));
}
