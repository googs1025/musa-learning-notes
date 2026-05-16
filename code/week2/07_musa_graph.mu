// ============================================================================
//  文件:07_musa_graph.mu
//  标题:Week 2 · 示例 7 · MUSA Graph · 把反复 launch 压成一次
//  目标:理解 launch overhead 在小 kernel 场景下的统治力
//        学会 Stream Capture → Graph → 反复 Launch 的最短路径
//  致敬:a-hamdi/GPU/tree/main(100 days of GPU)中关于 CUDA Graph
//        "10k 次 launch 对比"的实验设计;本例代码自写,概念照搬
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 7.1 ──────────────────────────────────────────────────────────────────
// │  launch overhead 是定值,小 kernel 场景下会主导
// └──────────────────────────────────────────────────────────────────────────
//
//  Week 1 #06 测过:每次 kernel<<<>>> 启动的"入队 + driver bookkeeping"成本
//  在几 µs 到几十 µs 之间,跟 kernel 本身的工作量无关。
//
//  这在大 kernel 场景下可以忽略(kernel 跑 10 ms,overhead 占 1%),
//  但在以下场景会要命:
//    • LLM 推理逐 token 解码:每 token 一次 forward,每个 op 都是小 kernel
//    • 迭代求解器 / 物理仿真:每 step 几十次 launch,跑 10 万 step
//    • 强化学习的 inference 环节
//
//  对这些场景,launch 占的时间可能 > 真实 GPU 计算时间。

// ┌─ § 7.2 ──────────────────────────────────────────────────────────────────
// │  MUSA Graph 的核心思路
// └──────────────────────────────────────────────────────────────────────────
//
//  把一段 stream 操作(包括 kernel launch、memcpy、memset)**录制**成一个
//  DAG(graph),之后每次"重放"只用一次 musaGraphLaunch,不再走每次 launch
//  的完整入队路径。
//
//  最简的 Stream Capture 路径:
//
//      musaStream_t s;  musaStreamCreate(&s);
//
//      musaStreamBeginCapture(s, musaStreamCaptureModeGlobal);
//      kernelA<<<..., s>>>(...);                  // ← 不是真跑,在录
//      kernelB<<<..., s>>>(...);
//      ...
//      musaGraph_t graph;
//      musaStreamEndCapture(s, &graph);
//
//      musaGraphExec_t exec;
//      musaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
//
//      for (int i = 0; i < 10000; ++i) {
//          musaGraphLaunch(exec, s);              // ★ 真跑,代价 ≈ 单次 launch
//      }
//      musaStreamSynchronize(s);
//
//      musaGraphExecDestroy(exec);
//      musaGraphDestroy(graph);
//      musaStreamDestroy(s);

// ┌─ § 7.3 ──────────────────────────────────────────────────────────────────
// │  Graph 的好处不止是"少 launch 一次"
// └──────────────────────────────────────────────────────────────────────────
//
//  • driver 拿到整张 DAG 后可以做整体优化:依赖分析、合并小拷贝、复用 stream slot
//  • host 端只下发一次"执行整张图"指令,跨 PCIe 的命令字节数大幅下降
//  • exec 是 immutable 的,反复用就行,无 GC / 重新解析的代价
//
//  代价:
//    • 录制时 kernel 不能真跑(看不到结果),调试要先关掉 capture
//    • 结构改了(增加 op、改 launch config)就要重新 instantiate
//    • 参数能微调(musaGraphExecUpdate),但拓扑变化不行

// ┌─ § 7.4 ──────────────────────────────────────────────────────────────────
// │  本例的实验设计
// └──────────────────────────────────────────────────────────────────────────
//
//  极小 kernel:N=1024,逐元素加常数。每次跑 < 几 µs。
//
//  关键设计:**每个"逻辑 step" 内部跑 OPS_PER_STEP=5 个 kernel**,
//  模拟真实场景(LLM 推理一个 token 跑十几个 op、迭代求解一步多次 SAXPY)。
//  这样 Graph 才有"把多次 launch 合成一次"的舞台,单 kernel 的 graph 是反例。
//
//  两组:
//    A. 直接 launch:循环 ITERS 步,每步 5 次 kernel<<<>>>(共 5×ITERS launch)
//    B. 录制一次 5-kernel 序列 → 循环 ITERS 步 × musaGraphLaunch(共 ITERS 次)
//
//  用 GpuTimer 包整段,对比总时间。
//
//  注意:Graph 的实际收益 **强烈依赖驱动 / SDK 版本和 DAG 复杂度**。
//        如果你这次跑出来 B 比 A 慢,看 PART III Q1 的"实测可能反直觉"那段。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include "musa_common.h"
#include <cstdio>

__global__ void add_one(float* x, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
}

int main() {
    const int    N             = 1024;          // 故意小:让 launch overhead 显形
    const size_t bytes         = N * sizeof(float);
    const int    ITERS         = 5000;          // 步数
    const int    OPS_PER_STEP  = 5;             // 每步几个 kernel(模拟真实 DAG)
    const int    tpb           = 128;
    const int    bpg           = (N + tpb - 1) / tpb;

    float* d_x;
    MUSA_CHECK(musaMalloc(&d_x, bytes));

    musaStream_t s;
    MUSA_CHECK(musaStreamCreate(&s));

    // ── 跑法 A:直接循环 launch ──
    {
        MUSA_CHECK(musaMemsetAsync(d_x, 0, bytes, s));
        MUSA_CHECK(musaStreamSynchronize(s));

        GpuTimer t;
        t.start(s);
        for (int i = 0; i < ITERS; ++i) {
            for (int k = 0; k < OPS_PER_STEP; ++k) {
                add_one<<<bpg, tpb, 0, s>>>(d_x, N);
            }
        }
        t.stop(s);
        MUSA_CHECK(musaStreamSynchronize(s));
        int total = ITERS * OPS_PER_STEP;
        std::printf("[A] direct launch  %d steps × %d ops = %d launches : %7.3f ms  (%.2f µs/launch)\n",
                    ITERS, OPS_PER_STEP, total,
                    t.elapsed_ms(), t.elapsed_ms() * 1000.0 / total);
    }

    // ── 跑法 B:Stream Capture(把 5 个 kernel 录进一个 graph)→ 反复 Launch ──
    {
        MUSA_CHECK(musaMemsetAsync(d_x, 0, bytes, s));
        MUSA_CHECK(musaStreamSynchronize(s));

        MUSA_CHECK(musaStreamBeginCapture(s, musaStreamCaptureModeGlobal));
        for (int k = 0; k < OPS_PER_STEP; ++k) {
            add_one<<<bpg, tpb, 0, s>>>(d_x, N);
        }
        musaGraph_t graph;
        MUSA_CHECK(musaStreamEndCapture(s, &graph));

        musaGraphExec_t exec;
        MUSA_CHECK(musaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

        GpuTimer t;
        t.start(s);
        for (int i = 0; i < ITERS; ++i) {
            MUSA_CHECK(musaGraphLaunch(exec, s));
        }
        t.stop(s);
        MUSA_CHECK(musaStreamSynchronize(s));
        std::printf("[B] graph  launch  %d steps (each = %d ops in graph)        : %7.3f ms  (%.2f µs/step)\n",
                    ITERS, OPS_PER_STEP,
                    t.elapsed_ms(), t.elapsed_ms() * 1000.0 / ITERS);

        MUSA_CHECK(musaGraphExecDestroy(exec));
        MUSA_CHECK(musaGraphDestroy(graph));
    }

    // ── 验证 ──
    //    A 后 d_x = ITERS*OPS,B 又从 0 累加 ITERS*OPS,结果 = ITERS*OPS
    int expect = ITERS * OPS_PER_STEP;
    float h_x0;
    MUSA_CHECK(musaMemcpy(&h_x0, d_x, sizeof(float), musaMemcpyDeviceToHost));
    std::printf("verify x[0] = %.1f   (expect %d)   %s\n",
                h_x0, expect, h_x0 == static_cast<float>(expect) ? "OK ✓" : "FAILED ✗");

    MUSA_CHECK(musaStreamDestroy(s));
    MUSA_CHECK(musaFree(d_x));
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: B 比 A 快多少?(实测可能反直觉)
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 理论上,Graph 的收益来自"把 OPS_PER_STEP 次 launch 合成一次 launch":
//      A: ITERS × OPS launches × per_launch_cost
//      B: ITERS × 1 graph_launch_cost
//    只要 graph_launch_cost < OPS × per_launch_cost,Graph 就赢。
//
//  ▸ **实测可能不赢甚至更慢**——MUSA Graph 的 launch 路径在不同 SDK 版本
//    上成熟度不一样:
//      • 早期 SDK 上,musaGraphLaunch 自身的 driver 路径开销可能大于
//        几次直接 launch 的总和,单 kernel 甚至 5 kernel 的 graph 都不
//        赚反亏
//      • 直接 launch 是 driver 长期优化过的热路径,Graph 是相对新的特性
//    本仓库实测时,MTT S4000 上 5-op-per-graph 仍可能慢于直接 launch。
//
//  ▸ 即使如此,**Graph 的 API 本身值得掌握**:
//      • 比较新的 SDK 大概率会改进
//      • CUDA 上 Graph 已经是 LLM 推理的常用优化(Graph 包整个 layer)
//      • 你需要知道 Capture / Instantiate / Launch 这套范式,
//        免得未来 SDK 优化到位时不会用
//
//  // TODO: AutoDL 跑通后回填本仓库实测 ms 和 ratio,把"现状"写清楚

// ★ Q2: Graph 录制阶段 kernel 不真跑,那 add_one 内部访问的 d_x 不会真被改吧?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 对,capture 期间 kernel 不真跑,d_x 不会变。这是常见的"我录完跑出来
//    结果是 0"踩坑点:你以为录的时候算了一次,其实没算。
//
//  ▸ 验证方法:本例在 B 跑之前手动 musaMemsetAsync 把 d_x 清零,
//    然后只靠 musaGraphLaunch × ITERS 累加。最后 x[0] = ITERS 才是对的。
//
//  ▸ 实际工程里:capture 一次,后续大量 launch 输入相同结构的数据(只换 d_x
//    指针内容,不改 launch config),最适合静态 DAG 的推理 / 迭代场景。

// ★ Q3 (扩展): Graph 能让 launch overhead 降到 0 吗?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 不能。musaGraphLaunch 本身也有一次"提交命令给 GPU"的开销,
//    只是比"每次 launch 都要解析参数 + push 进 stream"轻得多。
//
//  ▸ 极致优化(论文里见过):把多次 musaGraphLaunch 也合并——
//    比如用 musaGraphAddKernelNode 把循环 N 次的逻辑展开成 N 个 node,
//    一次 musaGraphLaunch 完成整批。但 N 太大时 graph 实例化代价上升,
//    要权衡。
// ============================================================================
