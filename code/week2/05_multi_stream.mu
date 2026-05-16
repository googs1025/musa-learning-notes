// ============================================================================
//  文件:05_multi_stream.mu
//  标题:Week 2 · 示例 5 · 多流流水线 · 把 H2D / Kernel / D2H 重叠
//  目标:从"一切都串行"进化到"三件事可以同时干"
//        理解为什么真异步必须 pinned + Async + non-default stream
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 5.1 ──────────────────────────────────────────────────────────────────
// │  默认流(NULL stream)是串行的
// └──────────────────────────────────────────────────────────────────────────
//
//      musaMemcpy(d_a, h_a, ..., H2D);        // 默认流
//      kernel<<<...>>>(d_a);                  // 默认流
//      musaMemcpy(h_c, d_c, ..., D2H);        // 默认流
//
//  三件事在默认流上严格按提交顺序执行,任何时刻 GPU 上"只有一件事在跑"。
//  GPU 上的拷贝引擎(DMA)和计算单元(SM)其实是独立硬件,这种写法浪费了硬件。

// ┌─ § 5.2 ──────────────────────────────────────────────────────────────────
// │  多流就能让硬件并发跑
// └──────────────────────────────────────────────────────────────────────────
//
//      musaStream_t s1, s2;
//      musaStreamCreate(&s1);
//      musaStreamCreate(&s2);
//
//      musaMemcpyAsync(..., s1);              // 走拷贝引擎
//      kernel<<<g, b, 0, s2>>>(...);          // 走 SM
//                                             // ↑ 两件事可以同时!
//
//  GPU 调度器看到 s1 / s2 没依赖、占的硬件不冲突,就并发派发。
//
//  注意 kernel 启动语法多了两个参数:`<<<grid, block, sharedBytes, stream>>>`,
//  sharedBytes 是动态 shared memory 大小,本例不用,填 0。

// ┌─ § 5.3 ──────────────────────────────────────────────────────────────────
// │  真异步的三个必要条件
// └──────────────────────────────────────────────────────────────────────────
//
//      ① host 内存是 pinned (musaMallocHost,见 #02)
//      ② 用 musaMemcpyAsync,而不是 musaMemcpy
//      ③ 在 non-default stream 上(默认流 = 同步点)
//
//  少一个就退化:
//    • 用 pageable + Async → driver 在内部退化成同步,Async 名字骗人
//    • Pinned + Async + 默认流 → 默认流自带同步,失去并发
//    • Pinned + Memcpy(非 Async) → 函数本身阻塞 host

// ┌─ § 5.4 ──────────────────────────────────────────────────────────────────
// │  流水线模式
// └──────────────────────────────────────────────────────────────────────────
//
//  把 N 切成 K 份 chunk,每份用一个 stream 跑完整三段:
//
//      stream 0 :  [H2D₀]→[K₀]→[D2H₀]
//      stream 1 :          [H2D₁]→[K₁]→[D2H₁]
//      stream 2 :                  [H2D₂]→[K₂]→[D2H₂]
//      ...
//
//  GPU 上拷贝引擎(2 个方向独立)+ SM 可以三件事同时跑。
//  总耗时理论下降到 ≈ max(总 H2D, 总 K, 总 D2H),而不是三者之和。
//
//  本例切 4 chunk,4 个 stream 轮转,对照单流串行版,看总时间差距。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include "musa_common.h"
#include <cstdio>
#include <cmath>

__global__ void vector_add(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

int main() {
    const int    N      = 1 << 24;          // 16M elements, 64 MB
    const size_t bytes  = N * sizeof(float);
    const int    CHUNKS = 4;
    const int    n      = N / CHUNKS;
    const size_t cbytes = bytes / CHUNKS;

    // ── pinned host memory(真异步必须)──
    float *h_A, *h_B, *h_C;
    MUSA_CHECK(musaMallocHost(&h_A, bytes));
    MUSA_CHECK(musaMallocHost(&h_B, bytes));
    MUSA_CHECK(musaMallocHost(&h_C, bytes));
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = 2.0f * i;
    }

    // ── device 端 buffer(一次性大的,不切)──
    float *d_A, *d_B, *d_C;
    MUSA_CHECK(musaMalloc(&d_A, bytes));
    MUSA_CHECK(musaMalloc(&d_B, bytes));
    MUSA_CHECK(musaMalloc(&d_C, bytes));

    const int tpb = 256;
    const int bpg = (n + tpb - 1) / tpb;

    // ── 跑法 A:单流(默认流)串行 ──
    {
        GpuTimer t;
        t.start();
        MUSA_CHECK(musaMemcpy(d_A, h_A, bytes, musaMemcpyHostToDevice));
        MUSA_CHECK(musaMemcpy(d_B, h_B, bytes, musaMemcpyHostToDevice));
        vector_add<<<(N+tpb-1)/tpb, tpb>>>(d_A, d_B, d_C, N);
        MUSA_CHECK(musaMemcpy(h_C, d_C, bytes, musaMemcpyDeviceToHost));
        t.stop();
        std::printf("[A] single stream serial : %7.3f ms\n", t.elapsed_ms());
    }

    // ── 跑法 B:多流流水线 ──
    {
        musaStream_t streams[CHUNKS];
        for (int i = 0; i < CHUNKS; ++i) {
            MUSA_CHECK(musaStreamCreate(&streams[i]));
        }

        GpuTimer t;
        t.start();
        for (int c = 0; c < CHUNKS; ++c) {
            size_t off = c * n;
            musaStream_t s = streams[c];

            MUSA_CHECK(musaMemcpyAsync(d_A + off, h_A + off, cbytes,
                                       musaMemcpyHostToDevice, s));
            MUSA_CHECK(musaMemcpyAsync(d_B + off, h_B + off, cbytes,
                                       musaMemcpyHostToDevice, s));
            vector_add<<<bpg, tpb, 0, s>>>(d_A + off, d_B + off, d_C + off, n);
            MUSA_CHECK(musaMemcpyAsync(h_C + off, d_C + off, cbytes,
                                       musaMemcpyDeviceToHost, s));
        }
        // 等所有流完成
        for (int i = 0; i < CHUNKS; ++i) {
            MUSA_CHECK(musaStreamSynchronize(streams[i]));
        }
        t.stop();
        std::printf("[B] %d-stream pipeline    : %7.3f ms\n", CHUNKS, t.elapsed_ms());

        for (int i = 0; i < CHUNKS; ++i) {
            MUSA_CHECK(musaStreamDestroy(streams[i]));
        }
    }

    // ── 验证 ──
    int bad = 0;
    for (int i = 0; i < N; ++i) {
        if (std::fabs(h_C[i] - (h_A[i] + h_B[i])) > 1e-5f) { ++bad; if (bad>5) break; }
    }
    std::printf("verify: %s\n", bad == 0 ? "OK ✓" : "FAILED ✗");

    MUSA_CHECK(musaFree(d_A));
    MUSA_CHECK(musaFree(d_B));
    MUSA_CHECK(musaFree(d_C));
    MUSA_CHECK(musaFreeHost(h_A));
    MUSA_CHECK(musaFreeHost(h_B));
    MUSA_CHECK(musaFreeHost(h_C));
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 把 musaMallocHost 换成 malloc,会怎样?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ musaMemcpyAsync 看到 pageable host 内存,在 driver 内部退化为同步路径:
//    每次都是"先 memcpy 到内部 pinned staging,再 DMA",而且整个流程对 host
//    线程是阻塞的。
//
//  ▸ 结果:[B] 跑出来的耗时几乎跟 [A] 一样,4 个 stream 退化成 1 个串行流水线。
//
//  ▸ 这就是"Async API 用 pageable 内存等于自欺欺人"这条经验的来源。
//
//  // TODO: AutoDL 跑通后实测对比"pinned vs pageable + Async"两版的总耗时

// ★ Q2: CHUNKS 应该选几?太多 / 太少各什么后果?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 太少(CHUNKS=1):退化成单流,失去并发。
//  ▸ CHUNKS=2:基本能让 H2D 和 kernel 重叠,但 D2H 还得等。
//  ▸ CHUNKS=3-4:H2D/Kernel/D2H 三件事都能填上 GPU 硬件。
//  ▸ 太多(CHUNKS>>8):每块数据太小,launch overhead 主导,反而变慢。
//
//  ▸ 经验值:stream 数 ≥ 3,chunk 大小让单次 kernel 至少跑 100µs 以上。
//          对小数据用 #07 Graph 才合适(把 launch 开销压平)。

// ★ Q3 (扩展): 为什么 device 端 buffer 不切?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ device 端只要一块大 buffer,各 chunk 写入不同 offset。优势:
//      • 显存连续,kernel 访问是合并的
//      • 不用维护"chunk_idx → device_ptr_idx"的映射
//
//  ▸ host 端也可以这么做(本例就是),把 h_A 当成一个大 staging,
//    musaMemcpyAsync 直接给 offset 指针。
//
//  ▸ 如果未来 host 数据是从网络 / 文件**陆续到达**的(典型:推理服务),
//    那 host 端反而要切成 per-chunk buffer,每块一到位就立刻入流,
//    这就是 prefetch pipeline 的另一个变种。
// ============================================================================
