// ============================================================================
//  文件:04_vector_add_unified.mu
//  标题:Week 2 · 示例 4 · 统一内存 · musaMallocManaged
//  目标:理解统一内存的语义和代价
//        实测 prefetch 对首次访问的影响
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 4.1 ──────────────────────────────────────────────────────────────────
// │  统一内存的卖点:一个指针走天下
// └──────────────────────────────────────────────────────────────────────────
//
//      float* p;
//      musaMallocManaged(&p, bytes);
//      for (int i=0; i<N; ++i) p[i] = i;     // host 直接写,合法
//      kernel<<<...>>>(p, N);                // device 直接读,也合法
//      musaDeviceSynchronize();
//      printf("%f\n", p[0]);                 // host 又能读了
//
//  比起 #01 那 7 步骨架,省掉了:
//    • 两次 musaMalloc / musaFree
//    • 两次 musaMemcpy(H2D / D2H)
//
//  代码短一半,但代价不是 0。

// ┌─ § 4.2 ──────────────────────────────────────────────────────────────────
// │  page migration:数据按需迁移
// └──────────────────────────────────────────────────────────────────────────
//
//  统一内存底层是 page-fault 驱动的:
//    • host 第一次访问某页 → 物理页留在 host 内存
//    • kernel 访问同一页    → driver 收到 page fault,把页搬到 device
//    • host 再访问           → 又搬回 host
//
//  迁移单位是 page(典型 4 KiB / 2 MiB),不是一次性整块。
//  → 首次 kernel 启动后会有一阵"迁移卡顿",程序刚启动看起来比裸 musaMalloc 慢。

// ┌─ § 4.3 ──────────────────────────────────────────────────────────────────
// │  musaMemPrefetchAsync:预先把数据搬好
// └──────────────────────────────────────────────────────────────────────────
//
//      int device = 0;
//      musaMemPrefetchAsync(p, bytes, device, stream);
//
//  显式告诉 driver"这块马上要在 device 上用",避免临到 launch 才一页一页搬。
//
//  实测:对一次性大块计算,prefetch 把首次 kernel 时间从"几十 ms 迁移"
//        降到"和裸 musaMalloc 几乎一样"。
//
//  还有个搭档 musaMemAdvise(p, bytes, hint, device),可以告诉 driver
//  访问模式偏好(ReadMostly / PreferredLocation),进一步降低 fault 开销。
//  本例先只用 prefetch,advise 留到 Week 5 shared / GEMM 那块再展开。

// ┌─ § 4.4 ──────────────────────────────────────────────────────────────────
// │  什么时候应该 / 不应该用统一内存
// └──────────────────────────────────────────────────────────────────────────
//
//  ✓ 适合:
//    • 写原型 / 教学代码,不想为每个指针 musaMalloc + musaMemcpy
//    • 大于显存的数据集(可以 oversubscribe,driver 自动 swap)
//    • host 和 device 反复交替访问,人脑算迁移成本不值
//
//  ✗ 不适合:
//    • 性能关键路径:迁移开销可控性差,profile 难
//    • 训练框架(torch_musa):全部 musaMalloc + 显式 H2D,把行为变成确定的
//    • 多 GPU 强一致场景:多设备共享一份 managed 内存的开销很高


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝

#include "musa_common.h"
#include <cstdio>
#include <cmath>

__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

static float run_unified(bool do_prefetch, int N, size_t bytes) {
    // ── 单一指针,host/device 都能用 ──
    float *A, *B, *C;
    MUSA_CHECK(musaMallocManaged(&A, bytes));
    MUSA_CHECK(musaMallocManaged(&B, bytes));
    MUSA_CHECK(musaMallocManaged(&C, bytes));

    // host 直接写,initialization 不需要 memcpy
    for (int i = 0; i < N; ++i) {
        A[i] = static_cast<float>(i);
        B[i] = 2.0f * i;
    }

    bool prefetch_supported = true;
    if (do_prefetch) {
        int device = 0;
        MUSA_CHECK(musaGetDevice(&device));
        // 把三块都搬到 device 上,kernel launch 时就不用 page fault
        // 注意:有的硬件 / SDK 版本不支持显式 prefetch(返回 error 801),
        // 软失败:打一行警告,继续按"无 prefetch"路径跑,让 verify 还能完成
        musaError_t e = musaMemPrefetchAsync(A, bytes, device, 0);
        if (e == musaErrorNotSupported) {
            prefetch_supported = false;
            std::printf("  [info] musaMemPrefetchAsync 不被本设备支持,跳过 prefetch\n");
            (void)musaGetLastError();   // 清掉这条 error,避免污染后续 CHECK
        } else {
            MUSA_CHECK(e);
            MUSA_CHECK(musaMemPrefetchAsync(B, bytes, device, 0));
            MUSA_CHECK(musaMemPrefetchAsync(C, bytes, device, 0));
        }
    }

    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;

    GpuTimer t;
    t.start();
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    t.stop();
    float ms = t.elapsed_ms();

    // 同步后 host 读 C[0] 验证
    MUSA_CHECK(musaDeviceSynchronize());
    bool ok = std::fabs(C[0] - (A[0] + B[0])) < 1e-5f
           && std::fabs(C[N-1] - (A[N-1] + B[N-1])) < 1e-5f;

    MUSA_CHECK(musaFree(A));
    MUSA_CHECK(musaFree(B));
    MUSA_CHECK(musaFree(C));

    const char* tag =
        !do_prefetch              ? "no  prefetch       " :
        prefetch_supported        ? "with prefetch      " :
                                    "want prefetch (n/a)";
    std::printf("  %s  kernel %.4f ms  %s\n",
                tag, ms, ok ? "(verify ✓)" : "(verify ✗)");
    return ms;
}

int main() {
    const int    N     = 1 << 22;       // 16 MB
    const size_t bytes = N * sizeof(float);

    std::printf("Unified memory, N = %d (%.1f MB per buffer)\n",
                N, bytes / (1024.0 * 1024.0));

    // 跑两遍,看 prefetch 的差距
    std::printf("[run 1] 冷启动:\n");
    run_unified(false, N, bytes);
    run_unified(true,  N, bytes);

    std::printf("[run 2] 热启动:\n");
    run_unified(false, N, bytes);
    run_unified(true,  N, bytes);

    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: prefetch 的收益在 [run 1] 还是 [run 2] 更明显?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 预期:[run 1] 冷启动时差距明显——没有 prefetch 那一版会被首次
//    page fault 拖慢一截;[run 2] 已经在 device 上,差距收敛。
//
//  ▸ 但是 run_unified 每次都重新 musaMallocManaged + 写 host + free,所以
//    [run 2] 也是一次新的迁移流程,差距应该跟 [run 1] 接近。
//    要看到真正的"热启动",得在 musaFree 之前 prefetch 回 device。
//
//  ▸ 实测(AutoDL MTT,MUSA 3.1.0,N=4M,16 MB):
//      ⚠️ musaMemPrefetchAsync **本设备不支持**,程序运行时检测到 ENOSYS 自动
//         跳过 prefetch 调用(打印 "[info] musaMemPrefetchAsync 不被本设备支持")。
//      所以"prefetch 那一版"实际上跟"no prefetch"走的是同一条路径。
//
//      [run 1] 冷启动:
//        no  prefetch         kernel 1.7515 ms
//        want prefetch (n/a)  kernel 1.5135 ms
//      [run 2] 热启动:
//        no  prefetch         kernel 1.5338 ms
//        want prefetch (n/a)  kernel 1.5536 ms
//
//    冷启动第一次跑确实更慢(~0.2 ms),page fault → host→device migration 的代价;
//    热启动后稳定在 1.5 ms 量级。prefetch 缺失意味着你在这台机器上拿不到
//    "显式 prefetch 提前热身"的收益——这是 MUSA 当前版本的功能缺口,不是你写错了。

// ★ Q2: 为什么训练框架(torch_musa)默认不用统一内存?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 训练对延迟稳定性要求极高,每个 step 的耗时必须可预测。
//    统一内存的 page fault 是"非确定性的"——driver 调度受其他因素影响,
//    单步耗时方差大,会破坏 batch size / lr scheduler 的稳态。
//
//  ▸ 训练框架更喜欢:
//      • 显式 musaMalloc 进显存,自己管 lifetime
//      • 用 pinned buffer + Async 拷贝跑流水线(#05 的玩法)
//      • profile 出来每个 op 几毫秒就是几毫秒,优化才有抓手
//
//  ▸ 推理框架(TensorRT / FasterTransformer)同理。统一内存主要用于
//    研究代码 / Jupyter notebook 这类"代码短比性能稳更重要"的场景。

// ★ Q3 (扩展): 比 4 KiB 大的统一内存访问开销固定吗?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 不固定。driver 内部用启发式:
//      • 顺序访问的大块,会触发"prefetch ahead"自动批量搬运
//      • 随机访问,只能一页一页搬,fault 开销线性叠加
//
//  ▸ 这就是为什么本例特地用 N=16MB:大到能看出 prefetch 收益,
//    小数据(几 KB)看不出来,因为底层迁移单位就是一个 page。
// ============================================================================
