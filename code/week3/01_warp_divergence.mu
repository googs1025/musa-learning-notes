// 预计输出：
//   coherent=... ms divergent=... ms slowdown=...x
// 注意：divergent 通常慢于 coherent，具体 slowdown 随硬件和编译优化变化。

#include "musa_common.h"
#include <cstdio>

__global__ void branch_kernel(float* out, int n, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = float(i & 255);
    if (mode == 0) {
        // coherent 分支：blockIdx.x 对一个 block 内的所有线程都相同。
        // 因此同一个 warp/group 里的线程会一起走 if 或一起走 else，
        // 不会因为这个条件产生分支发散。两边故意使用相同算式，
        // 这里只构造“控制流一致”的 baseline；由于两边算式相同，
        // 编译器也可能直接把这个内部 if 化简掉。
        if (blockIdx.x & 1) x = x * 1.0001f + 1.0f;
        else x = x * 1.0001f + 1.0f;
    } else {
        // divergent 分支：threadIdx.x 在同一个 warp/group 内各不相同。
        // 偶数 thread 走 else，奇数 thread 走 if；同一执行组需要
        // 分别执行两条路径，并暂时屏蔽不满足条件的 lane。
        // 这就是本例中 divergent 可能更慢的原因之一；注意两条路径
        // 还分别使用了乘法和除法，所以 slowdown 不是纯粹的 divergence 测量。
        if (threadIdx.x & 1) x = x * 1.0001f + 1.0f;
        else x = x / 1.0001f - 1.0f;
    }
    out[i] = x;
}

float run(float* d, int n, int mode) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    GpuTimer t;

    // 第一次 launch 只做 warm-up，不计入最终耗时：
    // 运行时可能在这里完成 context 初始化、模块加载和其他一次性准备。
    // 先执行并检查它，避免把这些冷启动开销混进 coherent/divergent 对比。
    branch_kernel<<<grid, block>>>(d, n, mode);
    MUSA_CHECK_KERNEL();

    // 正式测量连续执行 50 次，再除以 50 得到平均 kernel 耗时。
    // 多次重复可以降低单次调度抖动；如果要测“第一次启动延迟”，
    // 则应去掉上面的 warm-up，并单独定义测量目标。
    t.start();
    for (int r = 0; r < 50; ++r) branch_kernel<<<grid, block>>>(d, n, mode);
    t.stop();
    return t.elapsed_ms() / 50.0f;
}

int main() {
    const int N = 1 << 24;
    float* d = nullptr;
    MUSA_CHECK(musaMalloc(&d, N * sizeof(float)));
    // 两种情况
    float coherent = run(d, N, 0);
    float divergent = run(d, N, 1);
    std::printf("coherent=%.3f ms divergent=%.3f ms slowdown=%.2fx\n", coherent, divergent, divergent / coherent);
    MUSA_CHECK(musaFree(d));
}
