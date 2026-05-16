// ============================================================================
//  文件:02_vector_add_pinned.mu
//  标题:Week 2 · 示例 2 · Pinned Host Memory · 让 H2D 飞起来
//  目标:理解 pageable / pinned 内存的区别
//        实测 pinned 拷贝的加速比
//        为 #05 多流流水线打地基(真异步必须 pinned)
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 2.1 ──────────────────────────────────────────────────────────────────
// │  普通 malloc 的内存是 "pageable"
// └──────────────────────────────────────────────────────────────────────────
//
//  C 标准 malloc / new 分配的内存是 pageable 的:OS 可能随时把它换到磁盘,
//  物理地址也可能在 GC 时被搬来搬去。
//
//  GPU 的 DMA 引擎要的是稳定的物理地址,所以 H2D 时驱动得这么走:
//
//      pageable buffer(用户) ─copy→ pinned staging 区(driver 内部) ─DMA→ GPU
//
//  多了一次中转,带宽对半砍,延迟也增加。

// ┌─ § 2.2 ──────────────────────────────────────────────────────────────────
// │  pinned (page-locked) memory:跳过中转
// └──────────────────────────────────────────────────────────────────────────
//
//      musaMallocHost(&h_pinned, bytes);      // 经典 API
//      musaHostAlloc(&h_pinned, bytes, flags); // 带 flags,功能更多
//
//      // 用完释放:
//      musaFreeHost(h_pinned);
//
//  driver 把这块内存钉死在物理页里,GPU 可以直接 DMA 读写,链路就成了:
//
//      pinned buffer(用户) ─DMA→ GPU       ★ 一次拷贝,峰值带宽
//
//  代价:
//    • 物理页被钉住,OS 不能换出 → 不能开太大(吃光物理内存会拖累系统)
//    • 分配 / 释放比 malloc 慢 → 不适合频繁分配 / 释放小块
//    • flag = WriteCombined 可以更快(只写不读时),但 host 反向读会很慢
//
//  实践:训练 / 推理代码会在初始化时 musaMallocHost 一大块,反复复用。

// ┌─ § 2.3 ──────────────────────────────────────────────────────────────────
// │  pinned 是 "真异步" 的前提
// └──────────────────────────────────────────────────────────────────────────
//
//      musaMemcpyAsync(d, h, bytes, H2D, stream);
//
//  这个 API 名字叫 Async,但如果 h 是 pageable 内存,driver 没法 DMA,
//  只能退化成同步——表面 Async,实际还是阻塞。
//
//  真要让 H2D 跟 kernel 并发,必须三件事齐:
//    ① h 是 pinned (musaMallocHost)
//    ② 用 musaMemcpyAsync 而不是 musaMemcpy
//    ③ 用 non-default stream
//
//  → 这就是为什么 #05 multi_stream 必须先讲 pinned。

// ┌─ § 2.4 ──────────────────────────────────────────────────────────────────
// │  本例怎么测
// └──────────────────────────────────────────────────────────────────────────
//
//  同样 N、同样 bytes,只换 host 端分配方式:
//      A 组:malloc       → musaMemcpy H2D
//      B 组:musaMallocHost → musaMemcpy H2D
//
//  用 GpuTimer 包 musaMemcpy(GpuTimer 比 wall clock 更准、且不受 sync 影响),
//  跑 ITERS 次取后 ITERS-1 次的均值(丢首次冷启动)。
//  打印两组耗时 + 加速比。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

static float bench_h2d(float* h_src, float* d_dst, size_t bytes, int iters) {
    GpuTimer t;
    float total_ms = 0.0f;
    for (int it = 0; it < iters; ++it) {
        t.start();
        MUSA_CHECK(musaMemcpy(d_dst, h_src, bytes, musaMemcpyHostToDevice));
        t.stop();
        float ms = t.elapsed_ms();
        if (it > 0) total_ms += ms;     // 丢首次
    }
    return total_ms / (iters - 1);
}

int main() {
    const int    N     = 1 << 22;             // 16 MB,够大,带宽差异看得见
    const size_t bytes = N * sizeof(float);
    const int    ITERS = 6;                   // 实际取后 5 次均值

    // ── device buffer:两组共用 ──
    float* d_dst = nullptr;
    MUSA_CHECK(musaMalloc(&d_dst, bytes));

    // ── A 组:pageable (普通 malloc) ──
    float* h_pageable = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) h_pageable[i] = static_cast<float>(i);

    float avg_pageable = bench_h2d(h_pageable, d_dst, bytes, ITERS);

    // ── B 组:pinned (musaMallocHost) ──
    float* h_pinned = nullptr;
    MUSA_CHECK(musaMallocHost(&h_pinned, bytes));
    for (int i = 0; i < N; ++i) h_pinned[i] = static_cast<float>(i);

    float avg_pinned = bench_h2d(h_pinned, d_dst, bytes, ITERS);

    // ── 报告 ──
    double mb       = bytes / (1024.0 * 1024.0);
    double bw_page  = mb / (avg_pageable / 1000.0);   // MB/s 转 GB/s 自己除 1024
    double bw_pin   = mb / (avg_pinned   / 1000.0);
    std::printf("bytes        = %.1f MB   iters = %d (avg of last %d)\n",
                mb, ITERS, ITERS - 1);
    std::printf("pageable H2D : %7.3f ms   %.2f GB/s\n",
                avg_pageable, bw_page / 1024.0);
    std::printf("pinned   H2D : %7.3f ms   %.2f GB/s\n",
                avg_pinned,   bw_pin  / 1024.0);
    std::printf("speedup      : ×%.2f\n", avg_pageable / avg_pinned);

    // ── 清理 ──
    MUSA_CHECK(musaFreeHost(h_pinned));
    free(h_pageable);
    MUSA_CHECK(musaFree(d_dst));
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: pinned 的加速比预计多少?为什么 H2D 比 D2H 受益更明显?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 预计:×1.5 ~ ×3,具体看 PCIe 版本和系统配置。
//
//  ▸ 为什么 H2D 受益更明显:
//      • H2D 时驱动得提前把数据"凑"成 pinned,这一步要 CPU 干活
//      • D2H 反向:DMA 直接写到驱动的 pinned staging,再 memcpy 给用户的
//        pageable,也是一次额外拷贝,但只涉及 CPU memcpy 不需要锁页流程
//      • 实测两边收益级别接近,但 H2D 通常更显眼,因为它要先"准备"
//
//  // TODO: AutoDL 跑通后回填实测加速比

// ★ Q2: 那能不能把整个进程都用 pinned?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 不行。pinned 内存的物理页被钉死,OS 不能 swap、不能合并 / 整理。开太多会:
//      • 物理内存吃光,其他进程无 mem 可用
//      • OS 的 page reclaim 失效,系统响应变卡
//      • 极端情况触发 OOM killer
//
//  ▸ 实践规则:
//      • 训练:per-rank 几百 MB ~ 1-2 GB 的 pinned buffer 当 staging 是合理的
//      • 推理 / 流式数据:用 pinned ring buffer 反复复用,而不是 per-request 分配
//      • 不要把整个模型权重 musaMallocHost(直接 musaMalloc 进显存才对)

// ★ Q3 (扩展): WriteCombined 是什么?什么时候用?
// ──────────────────────────────────────────────────────────────────────────
//
//      musaHostAlloc(&p, bytes, musaHostAllocWriteCombined);
//
//  ▸ WriteCombined 内存禁用 CPU cache,host 端写入会合并成 burst,
//    H2D 时带宽可以再涨 10-30%。但是 host 端如果反向读这块内存会非常慢
//    (要走总线读 GPU 端缓存)。
//
//  ▸ 适合:host 只写、GPU 只读的 staging 区(典型:输入图像、prompt tokens)。
//          host 自己再算结果的 buffer 不要用 WriteCombined。
// ============================================================================
