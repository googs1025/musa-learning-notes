// ============================================================================
//  文件：06_async_kernel.mu
//  标题：Week 1 · 示例 6 · Kernel 启动是异步的
//  目标：用 wall clock 直观看到 kernel<<<>>> 立即返回
//        理解为什么"必须 musaDeviceSynchronize"
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 1.1 ──────────────────────────────────────────────────────────────────
// │  "异步"到底是什么意思
// └──────────────────────────────────────────────────────────────────────────
//
//      kernel<<<g, b>>>(args);
//      // ↑ 这一句几乎不耗时（只是把"启动请求"丢进队列）
//      // CPU 不会等 GPU 算完，立刻继续往下跑
//
//  好处：CPU 可以在 GPU 计算时同时做别的事（比如准备下一批数据），
//        天然形成 CPU/GPU 流水线，不浪费两边任何一方。
//
//  代价：CPU 拿不到结果！必须显式同步：
//        • musaDeviceSynchronize()    等所有流上的所有任务
//        • musaStreamSynchronize(s)   只等指定流
//        • musaMemcpy(D2H)            隐式同步（拷贝前内部 wait）
//        • musaEventSynchronize(e)    等某个事件触发

// ┌─ § 1.2 ──────────────────────────────────────────────────────────────────
// │  常见踩坑：忘了同步
// └──────────────────────────────────────────────────────────────────────────
//
//      kernel<<<>>>(d_out, ...);   // GPU 还在算
//      printf("%f\n", h_out[0]);   // ← 读的是脏数据，因为根本没拷回
//
//  正确写法：
//      kernel<<<>>>(d_out, ...);
//      musaMemcpy(h_out, d_out, ..., D2H);  // ← 这一行内部会等 kernel 跑完
//      printf("%f\n", h_out[0]);
//
//  → 所以日常代码里有时不用显式 musaDeviceSynchronize，
//    但**只要紧接着是 D2H 拷贝**，那次拷贝就替你做了同步。

// ┌─ § 1.3 ──────────────────────────────────────────────────────────────────
// │  GPU 计时不能用 CPU clock 直接量
// └──────────────────────────────────────────────────────────────────────────
//
//  错误示范：
//      auto t0 = chrono::now();
//      kernel<<<>>>(...);
//      auto t1 = chrono::now();
//      // t1-t0 ≈ launch 开销，根本不是 kernel 时间
//
//  正确做法：
//      (a) 简单粗暴 ──
//          kernel<<<>>>(...);
//          musaDeviceSynchronize();     // 强制等
//          // 现在 t1-t0 才有意义，但精度受限
//
//      (b) 精准 ── musaEvent（Week 2 介绍）
//          musaEvent t0,t1; musaEventCreate(...);
//          musaEventRecord(t0); kernel<<<>>>(...); musaEventRecord(t1);
//          musaEventSynchronize(t1);
//          musaEventElapsedTime(&ms, t0, t1);   // 毫秒级精度，GPU 时基

// ┌─ § 1.4 ──────────────────────────────────────────────────────────────────
// │  本例的设计
// └──────────────────────────────────────────────────────────────────────────
//
//  起一个"故意慢"的 kernel（每个线程跑长循环），
//  然后分别测：
//      • t_launch  ── kernel<<<>>>() 这一行耗时（应该极小）
//      • t_total   ── kernel<<<>>>() + musaDeviceSynchronize() 总耗时
//
//  正常情况下 t_launch ≪ t_total，差距越大说明 kernel 越重，
//  也越能直观感受到"launch 异步"。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
//  编译：make          运行：./06_async_kernel

#include <musa_runtime.h>
#include <chrono>
#include <cstdio>

#define CHECK(call) do {                                       \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        return 1;                                              \
    }                                                          \
} while (0)

// ── 故意慢的 kernel：每个线程跑 LOOPS 次空转 ──
//    用 volatile + 累加阻止编译器把循环优化掉。
__global__ void busy_kernel(int loops, int* sink) {
    volatile int acc = 0;
    for (int i = 0; i < loops; ++i) {
        acc += i;
    }
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        sink[0] = acc;        // 让结果有 side-effect，不会被优化掉
    }
}

static double ms_since(std::chrono::high_resolution_clock::time_point t0) {
    auto t1 = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int main() {
    int* d_sink = nullptr;
    CHECK(musaMalloc(&d_sink, sizeof(int)));

    const int LOOPS = 5'000'000;     // 故意调大，让 kernel 明显慢
    const int blocks  = 64;
    const int threads = 128;

    // ── 测 launch 本身耗时（不同步）──
    auto t0 = std::chrono::high_resolution_clock::now();
    busy_kernel<<<blocks, threads>>>(LOOPS, d_sink);
    double t_launch = ms_since(t0);

    // ── 现在阻塞等 GPU 跑完 ──
    auto t1 = std::chrono::high_resolution_clock::now();
    CHECK(musaDeviceSynchronize());
    double t_wait = ms_since(t1);

    // ── 总耗时（launch + 实际 GPU 计算）──
    double t_total = t_launch + t_wait;

    printf("kernel launch (返回时):  %.3f ms\n", t_launch);
    printf("等 GPU 跑完(sync 阶段):  %.3f ms\n", t_wait);
    printf("总耗时:                  %.3f ms\n", t_total);
    printf("\n");
    printf("观察：t_launch 通常是几十到几百微秒，\n");
    printf("      t_wait 才是 kernel 真正在 GPU 上的执行时间，\n");
    printf("      两者差距越大 = launch 异步性越明显。\n");

    CHECK(musaFree(d_sink));
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  练习题与解答（对应 exercises.md E1.9 / E1.10）               ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 把 LOOPS 调成 100，t_launch 和 t_wait 谁会变？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 思路：t_launch 只跟"启动开销"相关，跟 kernel 内部循环次数无关；
//          t_wait 跟实际 GPU 算多久成正比。
//
//  ▸ 答案：
//      • t_launch 几乎不变（依然是几十~几百微秒）
//      • t_wait 大幅缩短，可能跟 t_launch 差不多了
//      • 此时"launch 异步"的视觉效果消失 ── 因为 kernel 太短，
//        CPU 还没启动完，GPU 都已经跑完了
//
//  ▸ 启示：launch overhead 是定值（~微秒级），
//          所以"反复 launch 极小 kernel"的场景里它会成为瓶颈
//          → 这就是 MUSA Graph（Week 2 第 4 个示例）要解决的问题。

// ★ Q2: 把 musaDeviceSynchronize() 删掉，直接读 d_sink 会发生什么？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 答案：musaMemcpy(D2H) 是同步函数，内部会先等 default stream 上
//          所有任务完成，再开始拷贝。所以即使你删了 musaDeviceSynchronize，
//          只要后面跟着 D2H 的 musaMemcpy，结果依然正确。
//
//  ▸ 但是：直接 host 读 d_sink 是 segfault（详见 04_memory_basics 那个坑）。
//
//  ▸ 实务建议：
//      • 训练循环里最后一步如果就是 H←D 拷回 loss，可以省掉显式 sync
//      • 但**如果你要测 kernel 时间**，必须显式 sync，否则计的是 launch 时间

// ★ Q3 (扩展): 多个 kernel 排队会怎样？
// ──────────────────────────────────────────────────────────────────────────
//
//      busy_kernel<<<...>>>(...);    // launch 1
//      busy_kernel<<<...>>>(...);    // launch 2
//      busy_kernel<<<...>>>(...);    // launch 3
//      musaDeviceSynchronize();
//
//  ▸ 答案：3 次 launch 全部立即返回，三个任务进默认流后**顺序串行执行**，
//          最后 sync 等所有 3 个跑完。
//
//  ▸ 想真正并发？ → 用多个 stream（Week 2 第 3 个示例 03_multi_stream）。
// ============================================================================
