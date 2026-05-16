// ============================================================================
//  文件:01_vector_add_runtime.mu
//  标题:Week 2 · 示例 1 · vectorAdd · Runtime API 完整版
//  目标:把 GPU 程序的 "7 步骨架" 第一次完整跑一遍
//        建立 host/device 数据流的肌肉记忆
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 1.1 ──────────────────────────────────────────────────────────────────
// │  GPU 程序的"7 步骨架"
// └──────────────────────────────────────────────────────────────────────────
//
//      1. 算 bytes:    size_t bytes = N * sizeof(float);
//      2. host 分配:    float* h_x = (float*)malloc(bytes);
//      3. device 分配:  musaMalloc(&d_x, bytes);
//      4. H → D:        musaMemcpy(d_x, h_x, bytes, H2D);
//      5. kernel 计算:  vector_add<<<grid, block>>>(d_a, d_b, d_c, N);
//      6. D → H:        musaMemcpy(h_c, d_c, bytes, D2H);   ← 隐式同步
//      7. 释放:         musaFree(d_x);  free(h_x);
//
//  整个 Week 1 都在为这 7 步打地基,从本例开始正式拼起来。
//  后面所有计算 kernel 都是这套骨架的变体(加 stream、加 graph、加 unified)。

// ┌─ § 1.2 ──────────────────────────────────────────────────────────────────
// │  启动配置(grid / block)怎么选
// └──────────────────────────────────────────────────────────────────────────
//
//      blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
//      vector_add<<<blocksPerGrid, threadsPerBlock>>>(...);
//
//  threadsPerBlock 通常选 128 / 256 / 512:
//    • 太小(<64):block 数太多,SM 调度开销大,寄存器/shared 利用率低
//    • 太大(>512):每个 block 占的资源多,SM 上同时驻留的 block 少 → 占用率下降
//    • 256 是几乎所有教材的默认值,新手就先用 256
//
//  ceil 除法 `(N + t - 1) / t` 用来保证最后一个块覆盖余数。

// ┌─ § 1.3 ──────────────────────────────────────────────────────────────────
// │  边界保护 if (i < N) 为什么不能省
// └──────────────────────────────────────────────────────────────────────────
//
//  ceil 后实际启动的线程数往往 > N。比如 N=1023,threadsPerBlock=256:
//      blocksPerGrid = (1023 + 255) / 256 = 4   → 4 * 256 = 1024 个线程
//      多出来的 1 个线程算 i=1023,数组只有索引 0..1022
//      → 这 1 个线程如果不挡住,就会越界写,可能踩坏邻居内存(或 illegal address)
//
//  所以 kernel 里第一行几乎都是:
//      int i = blockIdx.x * blockDim.x + threadIdx.x;
//      if (i >= N) return;   // 或者 if (i < N) { ... }
//
//  → "线程数 ≥ 数据量,多余线程要 return" 是 GPU kernel 的通用模式。

// ┌─ § 1.4 ──────────────────────────────────────────────────────────────────
// │  正确性验证:host 端再算一遍
// └──────────────────────────────────────────────────────────────────────────
//
//  GPU 跑出来的结果别盲信,host 重算一份对比,差值 < 1e-5 (float) 才算过。
//  以后所有示例都用这种模式当冒烟测试,跑通了你才知道改 kernel 没改坏。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
//  编译:make 01_vector_add_runtime     运行:./01_vector_add_runtime

#include "musa_common.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ── Kernel:逐元素 C = A + B ───────────────────────────────────────────────
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // ── 步 1:N 和 bytes ──
    const int    N     = 1 << 20;          // 1,048,576 个 float ≈ 4 MB
    const size_t bytes = N * sizeof(float);

    // ── 步 2:host 分配 + 初始化 ──
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = 2.0f * i;
    }

    // ── 步 3:device 分配 ──
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    MUSA_CHECK(musaMalloc(&d_A, bytes));
    MUSA_CHECK(musaMalloc(&d_B, bytes));
    MUSA_CHECK(musaMalloc(&d_C, bytes));

    // ── 步 4:H → D ──
    MUSA_CHECK(musaMemcpy(d_A, h_A, bytes, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, bytes, musaMemcpyHostToDevice));

    // ── 步 5:kernel ──
    const int threadsPerBlock = 256;
    const int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    MUSA_CHECK_KERNEL();        // 同步 + 异步错误一次抓

    // ── 步 6:D → H(隐式同步,musaMemcpy 内部会等 kernel 跑完)──
    MUSA_CHECK(musaMemcpy(h_C, d_C, bytes, musaMemcpyDeviceToHost));

    // ── 验证 ──
    int  bad = 0;
    for (int i = 0; i < N; ++i) {
        float expect = h_A[i] + h_B[i];
        if (std::fabs(h_C[i] - expect) > 1e-5f) {
            if (bad < 5) {
                std::printf("mismatch @%d: got %f, expect %f\n",
                            i, h_C[i], expect);
            }
            ++bad;
        }
    }
    std::printf("vector_add N=%d  threadsPerBlock=%d  blocksPerGrid=%d\n",
                N, threadsPerBlock, blocksPerGrid);
    std::printf("%s\n", bad == 0 ? "OK ✓" : "FAILED ✗");

    // ── 步 7:释放 ──
    MUSA_CHECK(musaFree(d_A));
    MUSA_CHECK(musaFree(d_B));
    MUSA_CHECK(musaFree(d_C));
    free(h_A); free(h_B); free(h_C);
    return bad == 0 ? 0 : 1;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  Q&A                                                         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: threadsPerBlock 改成 1024 或 32 各会怎样?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 预期:
//      • 改成 1024:每个 block 占的资源多,同时驻留的 block 数下降。在 vectorAdd
//        这种纯访存 kernel 上耗时差异通常很小(瓶颈在带宽,不在调度)。
//      • 改成 32:block 数翻 8 倍,调度开销略增。MUSA 的 warp=128,32 个线程
//        连一个 warp 都填不满,SM 利用率不足。
//      • 改成 128(= 1 个 warp 整数倍):MUSA 上最自然的选择。
//
//  ▸ 启示:vectorAdd 这种带宽 bound 的 kernel,blocks/threads 调一两档影响不大,
//          但要养成"对齐 warp size"的习惯,等到 reduce / GEMM 才看得出差距。
//
//  ▸ 实测(AutoDL MTT,N=1M,threadsPerBlock=256,见 03_vector_add_timer 输出):
//      kernel 本身 0.119 ms(GpuTimer)
//      整段 main 包含 H2D + kernel + D2H 几 ms 级别,kernel 占小头
//    其他 threadsPerBlock 档位扫描留给 E2.1 习题。

// ★ Q2: 把 if (i < N) 删掉,N=1024 时正确吗?N=1023 时呢?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ N=1024,threadsPerBlock=256:blocksPerGrid=4 → 启动 1024 线程,
//    每个线程刚好一个元素,删了 if 也正确。
//
//  ▸ N=1023:blocksPerGrid 仍是 4 → 启动 1024 线程,多出 1 个线程算 i=1023。
//    这个线程会 C[1023] = A[1023] + B[1023] 越界写。
//    后果:可能命中相邻 musaMalloc 的另一块,污染下游;
//          也可能命中"未映射页",触发 illegal address,kernel 整体崩。
//
//  ▸ 启示:N 是 threadsPerBlock 整数倍时省 if 看着没事,但一改 N 就炸。
//          统一加 if 保护是零成本(分支预测命中率近 100%),不要省。

// ★ Q3 (扩展): 为什么 musaMemcpy(D2H) 之后 host 上 h_C 就能直接读?
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ musaMemcpy 默认是同步函数:发起 D2H 拷贝之前,driver 会先 wait default
//    stream 上所有任务(也就是上面 launch 的 vector_add)完成,才开始拷。
//    回到 host 这一行的时候,数据已经在 h_C 里了。
//
//  ▸ 这就是 Week 1 #06 讲过的"D2H 隐式同步":日常代码里不用再显式 sync,
//    但你要测 kernel 时间(03_vector_add_timer.mu)就必须自己加 sync,
//    因为那时候你想量的是 kernel 本身,而不是 launch+sync 的总和。
// ============================================================================
