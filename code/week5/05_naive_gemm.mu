// ============================================================================
//  文件：01_naive_gemm.mu
//  标题：Week 3 · 示例 1 · 朴素矩阵乘
//  目标：实现最简单的 GEMM，理解访存浪费在哪里
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 1.1 ──────────────────────────────────────────────────────────────────
// │  GEMM 的数学定义
// └──────────────────────────────────────────────────────────────────────────
//
//      C[M×N] = A[M×K] × B[K×N]
//
//      C[row, col] = Σ A[row, k] * B[k, col]   (k = 0..K-1)
//
//  浮点运算总数：2 * M * N * K（每个 C 元素 K 次乘加 = 2K flops）。

// ┌─ § 1.2 ──────────────────────────────────────────────────────────────────
// │  朴素并行：每线程算一个 C 元素
// └──────────────────────────────────────────────────────────────────────────
//
//  最直接的并行策略：
//    • 用 2D grid，gx = col, gy = row
//    • 每个线程独立读 A 的一整行 + B 的一整列，做内积
//
//  代码骨架：
//      int row = blockIdx.y * blockDim.y + threadIdx.y;
//      int col = blockIdx.x * blockDim.x + threadIdx.x;
//      float sum = 0;
//      for (int k = 0; k < K; ++k) sum += A[row*K+k] * B[k*N+col];
//      C[row*N+col] = sum;

// ┌─ § 1.3 ──────────────────────────────────────────────────────────────────
// │  为什么慢：访存严重浪费
// └──────────────────────────────────────────────────────────────────────────
//
//  设 block 是 16×16，每个 block 算 C 的 16×16 = 256 个元素。
//  这个 block 内：
//    • 用到 A 的 16 行（共 16*K 个元素）
//    • 用到 B 的 16 列（共 K*16 个元素）
//
//  但每个线程**独立**从 global memory 读，
//    A 的同一行被 16 个线程各自读一遍 → 浪费 16 倍带宽
//    B 的同一列也是 → 浪费 16 倍带宽
//
//  → 朴素 GEMM 是"重复读"的灾难。下个示例 02_tiled_gemm 通过
//    shared memory 把这种重复降到 1/TS。

// ┌─ § 1.4 ──────────────────────────────────────────────────────────────────
// │  Compute / Memory Roofline
// └──────────────────────────────────────────────────────────────────────────
//
//  Arithmetic Intensity (AI) = flops / bytes_read
//    朴素 GEMM 每个元素：2K flops，2K*4 bytes = 8K bytes → AI = 0.25 flops/byte
//    Tiled  GEMM (TS=16): 同样 2K flops，2K/TS * 4 bytes → AI ≈ 4 flops/byte
//
//  AI 越高 → 越能撑到算力顶；AI 越低 → 卡在带宽。
//  这就是 GPU 优化的核心思想：提高 AI。

// ┌─ § 1.5 ──────────────────────────────────────────────────────────────────
// │  GFLOPS 的算法
// └──────────────────────────────────────────────────────────────────────────
//
//      GFLOPS = (2 * M * N * K) / time_seconds / 1e9
//             = (2 * M * N * K) / (time_ms * 1e6)


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
//  编译：make          运行：./01_naive_gemm

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

#define MUSA_CHECK(call) do {                                       \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        std::exit(1);                                          \
    }                                                          \
} while (0)

// ── 朴素 GEMM kernel ──
__global__ void gemm_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.f;
        // ★ 每个线程要从 global 读 K 个 A + K 个 B（重复访问浪费带宽）
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytesA = M * K * sizeof(float);
    const size_t bytesB = K * N * sizeof(float);
    const size_t bytesC = M * N * sizeof(float);

    // host
    float *h_A = (float*)std::malloc(bytesA);
    float *h_B = (float*)std::malloc(bytesB);
    float *h_C = (float*)std::malloc(bytesC);
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.f;

    // device
    float *d_A, *d_B, *d_C;
    MUSA_CHECK(musaMalloc(&d_A, bytesA));
    MUSA_CHECK(musaMalloc(&d_B, bytesB));
    MUSA_CHECK(musaMalloc(&d_C, bytesC));
    MUSA_CHECK(musaMemcpy(d_A, h_A, bytesA, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, bytesB, musaMemcpyHostToDevice));

    // 2D launch
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    musaEvent_t t0, t1;
    musaEventCreate(&t0); musaEventCreate(&t1);

    // warmup（第一次 launch 包含 JIT/cache 加载，不算正式时间）
    gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    MUSA_CHECK(musaDeviceSynchronize());

    // 正式测时（多次取平均）
    musaEventRecord(t0);
    const int REPEAT = 10;
    for (int r = 0; r < REPEAT; ++r) {
        gemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    musaEventRecord(t1);
    musaEventSynchronize(t1);

    float ms = 0.f; musaEventElapsedTime(&ms, t0, t1);
    ms /= REPEAT;

    // GFLOPS = 2 * M * N * K / time
    double gflops = 2.0 * M * N * K / (ms * 1e6);

    MUSA_CHECK(musaMemcpy(h_C, d_C, bytesC, musaMemcpyDeviceToHost));
    printf("Naive GEMM  M=N=K=%d  time=%.2f ms  perf=%.1f GFLOPS  C[0]=%.0f (expect %.0f)\n",
           M, ms, gflops, h_C[0], (float)(K * 2));

    musaFree(d_A); musaFree(d_B); musaFree(d_C);
    std::free(h_A); std::free(h_B); std::free(h_C);
    musaEventDestroy(t0); musaEventDestroy(t1);
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  练习题与解答（对应 exercises.md E3.3）                       ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: 朴素 vs Tiled 加速比
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 跑这个版本和 02_tiled_gemm，记录两者 GFLOPS。
//
//  ▸ 典型差距（M=N=K=1024 时）：
//      Naive  ≈ 50 ~ 200 GFLOPS
//      Tiled  ≈ 500 ~ 2000 GFLOPS（看硬件）
//      加速比 ≈ 5 ~ 10x
//
//  ▸ 差距来源：
//      • Naive 每个 A/B 元素从 global 读 16 次（block size 维度）
//      • Tiled 用 shared memory 把这 16 次降到 1 次
//      → 等效带宽提升 16x，但还有别的瓶颈（指令、occupancy）所以只到 5~10x

// ★ Q2: 为什么实测 GFLOPS 远低于硬件 FP32 峰值？
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 答案：朴素版是 memory-bound。
//      AI = 0.25 flops/byte
//      实际带宽 ≈ 200 GB/s → 算力上限 = 200 * 0.25 = 50 GFLOPS
//      峰值 FP32 通常 > 10 TFLOPS → 利用率 < 1%
//
//  ▸ 验证方法：
//      1. 跑 03_saxpy_bandwidth 测实际带宽 BW
//      2. 期望 GFLOPS ≈ AI * BW
//      3. 实测应该接近这个值（在 memory-bound 阶段）

// ★ Q3 (扩展): row-major vs column-major
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 当前代码假设 A、B、C 都是 row-major (C 风格)。
//
//  ▸ 如果 B 改成 col-major（k 是 leading dim），访问变成：
//      B_col[k * N + col]  →  B_col[col * K + k]
//
//  ▸ 性能影响：
//      • 同 warp 32(/128) 个线程的 col 不同 → 跨步访问 B
//      • coalesced 度变差，性能掉 2~4x（见 Week 4）
//
//  ▸ 实战中 BLAS 接口默认 col-major（Fortran 传统），
//    cuBLAS / muBLAS 调用要注意 transpose 标志。
// ============================================================================
