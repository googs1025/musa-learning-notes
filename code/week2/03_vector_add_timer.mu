// ============================================================================
//  文件:03_vector_add_timer.mu
//  标题:Week 2 · 示例 3 · 正确量 kernel 时间 · musaEvent vs CPU clock
//  目标:理解为什么 CPU wall clock 量 kernel 几乎没用
//        学会用 GpuTimer (musaEvent) 量真实 kernel 时间
//        建立 warm-up + 多次取均值的基本方法学
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 3.1 ──────────────────────────────────────────────────────────────────
// │  复习:kernel 是异步的(Week 1 #06)
// └──────────────────────────────────────────────────────────────────────────
//
//      auto t0 = chrono::now();
//      kernel<<<...>>>(...);              // 几乎瞬间返回(只是把 launch 入队)
//      auto t1 = chrono::now();           // t1 - t0 ≈ launch overhead
//                                         // ★ 这不是 kernel 的真实时间!
//
//  → 想用 CPU 时钟量 kernel,必须先 musaDeviceSynchronize 等 GPU 跑完。
//    但即使这样,精度也只到毫秒级,且把 sync 等待时间也算进去了。

// ┌─ § 3.2 ──────────────────────────────────────────────────────────────────
// │  musaEvent:GPU 自己打的时间戳
// └──────────────────────────────────────────────────────────────────────────
//
//      musaEvent_t t0, t1;
//      musaEventCreate(&t0);
//      musaEventCreate(&t1);
//
//      musaEventRecord(t0, stream);
//      kernel<<<...>>>(...);
//      musaEventRecord(t1, stream);
//
//      musaEventSynchronize(t1);          // host 等 t1 真正发生
//      float ms;
//      musaEventElapsedTime(&ms, t0, t1); // 毫秒,~µs 精度
//
//  这就是 musa_common.h 里 GpuTimer 的实现。本例直接用 GpuTimer。
//
//  ★ musaEventRecord 不是"打时间戳",而是"在 stream 这个位置插一个标记,
//    GPU 调度到这里时记录硬件时间"。所以它的时基是 GPU 的、是异步的、
//    必须 musaEventSynchronize 等它真正发生才能读。

// ┌─ § 3.3 ──────────────────────────────────────────────────────────────────
// │  Warm-up 与多次取均值
// └──────────────────────────────────────────────────────────────────────────
//
//  GPU 首次 launch 比之后慢得多,常见原因:
//    • driver 还在做 module / context lazy init
//    • SM 频率从低功耗态拉满需要时间
//    • 第一次访问的内存还没缓存命中
//
//  实验方法学:
//    • 先 warm-up 1-2 次(结果丢掉)
//    • 跑 ITERS 次,记 min / avg / max
//    • 对小 kernel 还要"包起来跑多次再除"(放大单次耗时)
//
//  → 这是后面所有 benchmark 的标配,本例先用最简版本。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include "musa_common.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

int main() {
    const int    N     = 1 << 22;
    const size_t bytes = N * sizeof(float);

    // ── 数据准备(为聚焦计时,这里直接 device 上初始化省掉 H2D)──
    float *d_A, *d_B, *d_C;
    MUSA_CHECK(musaMalloc(&d_A, bytes));
    MUSA_CHECK(musaMalloc(&d_B, bytes));
    MUSA_CHECK(musaMalloc(&d_C, bytes));
    MUSA_CHECK(musaMemset(d_A, 0, bytes));      // 注意:Memset 按字节,这里只是图省事
    MUSA_CHECK(musaMemset(d_B, 0, bytes));

    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    const int WARMUP = 2;
    const int ITERS  = 10;

    // ── 跑法 A:CPU clock(不 sync)──
    //     一定会量错,目的是让你看到"几乎为 0"是怎么回事
    {
        CpuTimer t;
        t.start();
        vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        double ms = t.elapsed_ms();
        std::printf("[A] CPU clock, NO sync     : %8.4f ms   (量到的其实是 launch 入队耗时)\n", ms);
        MUSA_CHECK(musaDeviceSynchronize());    // 把 launch 的任务消化掉再继续
    }

    // ── 跑法 B:CPU clock + sync ──
    //     好歹包了 kernel 真实时间,但 sync 等待开销也混进来了,精度有限
    {
        // warm-up
        for (int i = 0; i < WARMUP; ++i) {
            vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        }
        MUSA_CHECK(musaDeviceSynchronize());

        double sum = 0.0, mn = 1e30, mx = 0.0;
        for (int i = 0; i < ITERS; ++i) {
            CpuTimer t;
            t.start();
            vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
            MUSA_CHECK(musaDeviceSynchronize());
            double ms = t.elapsed_ms();
            sum += ms; mn = std::min(mn, ms); mx = std::max(mx, ms);
        }
        std::printf("[B] CPU clock + sync       : avg %7.4f  min %7.4f  max %7.4f  (ms)\n",
                    sum / ITERS, mn, mx);
    }

    // ── 跑法 C:GpuTimer (musaEvent) ──
    //     这才是 kernel 在 GPU 上真实的执行时间
    {
        // warm-up
        for (int i = 0; i < WARMUP; ++i) {
            vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
        }
        MUSA_CHECK(musaDeviceSynchronize());

        float sum = 0.0f, mn = 1e30f, mx = 0.0f;
        for (int i = 0; i < ITERS; ++i) {
            GpuTimer g;
            g.start();
            vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
            g.stop();
            float ms = g.elapsed_ms();       // 内部会 musaEventSynchronize
            sum += ms; mn = std::min(mn, ms); mx = std::max(mx, ms);
        }
        std::printf("[C] GpuTimer (musaEvent)   : avg %7.4f  min %7.4f  max %7.4f  (ms)\n",
                    sum / ITERS, mn, mx);
    }

    MUSA_CHECK(musaFree(d_A));
    MUSA_CHECK(musaFree(d_B));
    MUSA_CHECK(musaFree(d_C));
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 为什么 [A] 跑法量出来几乎是 0?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ kernel<<<>>>(...) 这一行只把 launch 请求丢进队列就返回,真正的执行是
//    GPU 异步去跑的。CPU clock 在 launch 这一行 t.stop(),量到的就是
//    "入队 + driver bookkeeping" 这点开销,通常 0.01 ms 以下。
//
//  ▸ 但 kernel 在 GPU 上可能还要跑几毫秒——这部分根本没量上。
//
//  ▸ 这就是为什么任何"测 kernel 时间"的代码都必须先 sync(才知道 kernel 何时跑完)
//    或者用 musaEvent(GPU 自己打时间戳)。

// ★ Q2: [B] (CPU+sync) 和 [C] (musaEvent) 的差通常有多大?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 对单次几 ms 量级的 kernel,两者通常差几十 µs:
//      • [B] 包含 sync 等待 + driver 上下文切换 + chrono 调用开销
//      • [C] 是 GPU 内部时基,直接量 record 到 record 的硬件时间
//
//  ▸ 对极小 kernel(< 100 µs),[B] 可能比 [C] 大几倍(sync 开销主导)
//    这就是为什么基准测试一定要用 musaEvent / GpuTimer。
//
//  ▸ 实测(AutoDL MTT,N=1M,vectorAdd):
//      [A] CPU clock, NO sync     :  0.1554 ms   ← 量到的是 launch 入队耗时
//      [B] CPU clock + sync       :  0.1195 ms   (min 0.1180, max 0.1205)
//      [C] GpuTimer (musaEvent)   :  0.1224 ms   (min 0.1168, max 0.1495)
//    [B] vs [C] 差 ~3 µs(~2.4%),量级吻合,互相印证可信;
//    [A] 比真值小 30%(没 sync,只看到 launch 开销那点),典型"假数字"。
//    > kernel 越小,[B] 的 sync overhead 占比越大;真要测 µs 级 kernel,只能信 [C]。

// ★ Q3 (扩展): GpuTimer 在哪种情况下也会量错?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 跨 stream 不对齐:start 在 streamA、stop 在 streamB,GPU 之间没有
//    依赖关系,elapsed 可能是负的或乱跳。
//    → GpuTimer.start(s) / stop(s) 要给同一个 stream
//
//  ▸ 没真正 record:musaEventRecord 没在 kernel launch 前后,而是包到
//    更外层逻辑(比如包到 H2D 之前),那量的就不是 kernel 时间。
//
//  ▸ Graph(#07)里:Graph 内部的事件需要 musaGraphAddEventRecordNode,
//    简单 musaEventRecord 进不去 capture。这点 #07 注释会详细讲。
// ============================================================================
