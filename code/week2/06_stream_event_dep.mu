// ============================================================================
//  文件:06_stream_event_dep.mu
//  标题:Week 2 · 示例 6 · 跨流依赖 · event 当红绿灯
//  目标:把"两条流之间的有序"用 event 显式表达
//        理解 musaStreamWaitEvent 与 musaStreamSynchronize 的区别
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 6.1 ──────────────────────────────────────────────────────────────────
// │  同流强序、跨流无序
// └──────────────────────────────────────────────────────────────────────────
//
//      // 同一个 stream s1:
//      kernelA<<<..., s1>>>(...);
//      kernelB<<<..., s1>>>(...);   // ← B 一定在 A 之后跑(单 stream FIFO)
//
//      // 不同 stream s1 / s2:
//      kernelA<<<..., s1>>>(...);
//      kernelB<<<..., s2>>>(...);   // ← B 和 A 没任何依赖,可能同时跑
//                                   //   B 也可能比 A 先完成
//
//  → 多流让吞吐变高(#05),但也意味着"B 想用 A 的输出"这种依赖**必须显式声明**。

// ┌─ § 6.2 ──────────────────────────────────────────────────────────────────
// │  event 三步法
// └──────────────────────────────────────────────────────────────────────────
//
//  让 stream s2 上的 kernel 等 stream s1 上的 kernel 完成:
//
//      musaEvent_t e;
//      musaEventCreate(&e);
//
//      kernelA<<<..., s1>>>(...);
//      musaEventRecord(e, s1);                      // ① 在 s1 当前位置插标记
//
//      musaStreamWaitEvent(s2, e, 0);               // ② s2 等 e 触发再继续
//      kernelB<<<..., s2>>>(...);                   // ③ B 现在保证读到 A 的结果
//
//      // 用完释放
//      musaEventDestroy(e);
//
//  关键点:
//    • musaEventRecord(e, s1) **不是** "现在打时间戳",是"在 s1 当前位置
//      埋一个 marker,GPU 跑到这里时把 e 标为已触发"
//    • musaStreamWaitEvent(s2, e, 0) **不阻塞 host**,只是告诉 GPU:
//      "s2 的下一个任务,在 e 触发之前不要开始"
//    • 这是 device 上的同步,host 这行立刻返回

// ┌─ § 6.3 ──────────────────────────────────────────────────────────────────
// │  对比:musaStreamSynchronize 是 host 阻塞
// └──────────────────────────────────────────────────────────────────────────
//
//      musaStreamSynchronize(s1);    // host 在这里等 s1 跑完,丢失并发
//      kernelB<<<..., s2>>>(...);
//
//  跟 event 方案的区别:
//
//                       host 是否阻塞    s2 的其他任务能并发吗
//      Synchronize      ★ 是              ✗ host 卡住 → 整个 s2 都卡住
//      WaitEvent        ✗ 否              ✓ s2 早先入队的任务正常跑,
//                                            只是 e 之后的等
//
//  → 想 host 继续准备下一批数据,就用 event;反正都要等就 Synchronize。

// ┌─ § 6.4 ──────────────────────────────────────────────────────────────────
// │  本例的演示 DAG
// └──────────────────────────────────────────────────────────────────────────
//
//      s1:  fill_A(d_a)  ──record(e)──>
//                                       \
//      s2:                       wait(e) ──>  square_to_b(d_a → d_b)
//
//  fill_A 在 s1 上把 d_a[i] = i + 1 填好;
//  square_to_b 在 s2 上读 d_a 写 d_b[i] = d_a[i]²。
//  没 event 的话 square_to_b 可能在 fill_A 没跑完时启动,读到旧数据。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include "musa_common.h"
#include <cstdio>
#include <cmath>

__global__ void fill_A(float* a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = static_cast<float>(i + 1);
}

__global__ void square_to_b(const float* a, float* b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = a[i] * a[i];
}

int main() {
    const int    N     = 1 << 20;
    const size_t bytes = N * sizeof(float);
    const int    tpb   = 256;
    const int    bpg   = (N + tpb - 1) / tpb;

    float *d_a, *d_b;
    MUSA_CHECK(musaMalloc(&d_a, bytes));
    MUSA_CHECK(musaMalloc(&d_b, bytes));

    musaStream_t s1, s2;
    MUSA_CHECK(musaStreamCreate(&s1));
    MUSA_CHECK(musaStreamCreate(&s2));

    musaEvent_t a_done;
    MUSA_CHECK(musaEventCreate(&a_done));

    // ── s1 上跑 fill_A,record 一个事件 ──
    fill_A<<<bpg, tpb, 0, s1>>>(d_a, N);
    MUSA_CHECK(musaEventRecord(a_done, s1));

    // ── s2 上等 a_done,再跑 square_to_b ──
    MUSA_CHECK(musaStreamWaitEvent(s2, a_done, 0));
    square_to_b<<<bpg, tpb, 0, s2>>>(d_a, d_b, N);

    // ── 把结果拷回 host 验证(D2H 在 s2 上,自然等 square_to_b 完成)──
    float* h_b = (float*)malloc(bytes);
    MUSA_CHECK(musaMemcpyAsync(h_b, d_b, bytes, musaMemcpyDeviceToHost, s2));
    MUSA_CHECK(musaStreamSynchronize(s2));      // host 等流水线整体完成

    int bad = 0;
    for (int i = 0; i < N; ++i) {
        float expect = static_cast<float>(i + 1) * (i + 1);
        if (std::fabs(h_b[i] - expect) > 1e-5f * std::fabs(expect)) {
            if (bad < 5) std::printf("mismatch @%d: %f vs %f\n", i, h_b[i], expect);
            ++bad;
        }
    }
    std::printf("verify: %s   (bad = %d)\n", bad == 0 ? "OK ✓" : "FAILED ✗", bad);

    free(h_b);
    MUSA_CHECK(musaEventDestroy(a_done));
    MUSA_CHECK(musaStreamDestroy(s1));
    MUSA_CHECK(musaStreamDestroy(s2));
    MUSA_CHECK(musaFree(d_a));
    MUSA_CHECK(musaFree(d_b));
    return bad == 0 ? 0 : 1;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 删掉 musaStreamWaitEvent 那一行,会读到错的数据吗?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ "理论上"会,但实测在 vectorAdd 这种轻量 kernel 上不容易复现:
//      • s1 的 fill_A 太快,square_to_b 入队时它可能刚好已经跑完
//      • 单 GPU 上 driver 调度有一定串行性,看起来"碰巧对了"
//
//  ▸ 想可靠复现 race,把 fill_A 改慢(加 100 万次循环),再删 WaitEvent。
//    那时 square_to_b 几乎必然读到 0(d_a 默认 musaMalloc 没初始化的内容)。
//
//  ▸ 启示:GPU race 是"概率不可观察",代码看着对、跑 100 次都对、上线一变
//    数据规模就崩。生产代码 **任何跨流读 / 写都要显式 event**,不要赌运气。
//
//  // TODO: AutoDL 跑通后,把 fill_A 加 dummy 长循环,实测删 wait_event 的失败概率

// ★ Q2: 为什么不直接 musaStreamSynchronize(s1) 然后跑 s2?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 那是 host 阻塞:host 卡在 sync 这一行,不能再干别的(比如准备下一批输入)。
//    并发流水线的目的就是 host 一直在干活,GPU 也一直在干活——一旦 host
//    阻塞,流水线就断了。
//
//  ▸ event-based 等待是 device 上的等待,host 这一行 musaStreamWaitEvent
//    立刻返回,继续往下提交 s3 / s4 / 准备下一帧数据。GPU 调度器看到 s2 上
//    的任务依赖 event 才决定何时开跑。

// ★ Q3 (扩展): event 能跨 device / 跨进程吗?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 跨 device:可以,但 event 创建时要带 flag(musaEventInterprocess /
//    Disable Timing 等),用 musaIpcGetEventHandle 拿 handle 在另一进程
//    musaIpcOpenEventHandle 还原。
//
//  ▸ 跨进程:同上,基于 IPC handle。常见于多进程推理 server 之间 GPU 数据
//    共享,但每次创建有性能开销,通常不在 hot path 上。
//
//  ▸ 普通示例(本例)是单进程单 device,musaEventCreate 默认参数足够。
// ============================================================================
