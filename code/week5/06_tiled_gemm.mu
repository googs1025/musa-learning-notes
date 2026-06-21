// ============================================================================
//  文件：02_tiled_gemm.mu
//  标题：Week 3 · 示例 2 · 分块 GEMM（Shared Memory Tiling）
//  目标：用 shared memory 把每个 A/B 元素的 global 读次数降到 1/TS
// ============================================================================


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART I  ─  知识点                                                        ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ┌─ § 1.1 ──────────────────────────────────────────────────────────────────
// │  Shared Memory 是什么
// └──────────────────────────────────────────────────────────────────────────
//
//  位置：每个 SM 内部的 SRAM
//  大小：每 block 通常 48~128 KB（用 03_device_info 确认）
//  延迟：~20 周期（比 global 快 ~100 倍）
//  作用域：block 级。同 block 的所有线程共享，block 之间不可见
//  生命周期：随 block 启动分配、结束释放
//
//  声明方式：
//      __shared__ float buf[256];           // 静态大小
//      extern __shared__ float buf[];       // 动态大小，launch 时传

// ┌─ § 1.2 ──────────────────────────────────────────────────────────────────
// │  Tiling 的思想
// └──────────────────────────────────────────────────────────────────────────
//
//  把 C 切成 TS×TS 的小块，让一个 block 负责一个小块：
//
//      C[row..row+TS, col..col+TS] = Σ_t A_tile_t * B_tile_t
//
//  其中 A_tile_t / B_tile_t 是 K 方向的第 t 个 TS 宽切片。
//
//  关键：每次外层迭代，先把当前 tile 装到 shared，
//        然后 block 内 TS² 个线程都从 shared 取 → 复用率 = TS。

// ┌─ § 1.3 ──────────────────────────────────────────────────────────────────
// │  __syncthreads() 的两个时机
// └──────────────────────────────────────────────────────────────────────────
//
//  装载完 tile 后：等所有线程把数据搬到 shared 才开始计算
//      load(tileA, tileB);
//      __syncthreads();   // ★ 必须，否则有人在用脏数据
//
//  计算完一轮后：等所有线程算完，才能用 tile 装下一段
//      for (k=0..TS) sum += tileA[..] * tileB[..];
//      __syncthreads();   // ★ 必须，否则下一轮 load 会覆盖正在用的 tile
//
//  漏掉任一句 → 数据竞态、结果偶发错。

// ┌─ § 1.4 ──────────────────────────────────────────────────────────────────
// │  数据复用率的量化
// └──────────────────────────────────────────────────────────────────────────
//
//  Naive：每个 A/B 元素从 global 读 TS 次
//  Tiled：每个元素从 global 读 1 次，从 shared 读 TS 次
//
//  → global 带宽需求降为 1/TS。当 TS=16 → 16x 提升空间。

// ┌─ § 1.5 ──────────────────────────────────────────────────────────────────
// │  TS 的取值平衡
// └──────────────────────────────────────────────────────────────────────────
//
//  TS 太小（如 4）：复用率不够，shared memory 利用率低
//  TS 太大（如 64）：
//    • shared memory 占用过多 → 同 SM 上能跑的 block 数减少
//    • block 总线程数 = TS² 可能超过 maxThreadsPerBlock (1024)
//
//  常见折中：TS = 16 或 32。

// ┌─ § 1.6 ──────────────────────────────────────────────────────────────────
// │  Bank Conflict
// └──────────────────────────────────────────────────────────────────────────
//
//  Shared memory 被分成 32 个 bank（每个 4 字节宽）。
//  同 warp 32 个线程同时访问 → 如果落到同一 bank → 串行（慢）
//
//  典型避免：在 shared 数组的最内维加 +1 padding：
//      __shared__ float tileA[TS][TS+1];
//
//  当前 SDK 上不一定看得出明显差异（要用 profiler 验）。


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART II  ─  代码实现                                                     ║
// ╚══════════════════════════════════════════════════════════════════════════╝
//
//  编译：make          运行：./02_tiled_gemm

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

constexpr int TS = 16;   // tile size

__global__ void gemm_tiled(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    // ── 每 block 一份 shared，TS×TS 大小 ──
    __shared__ float tileA[TS][TS];
    __shared__ float tileB[TS][TS];

    int row = blockIdx.y * TS + threadIdx.y;
    int col = blockIdx.x * TS + threadIdx.x;

    float sum = 0.f;
    int numTiles = (K + TS - 1) / TS;

    // ── K 方向上分 numTiles 段，每段做一轮装载+计算 ──
    for (int t = 0; t < numTiles; ++t) {
        // 协同加载：每个线程负责把 1 个 A 元素 + 1 个 B 元素搬进 shared
        int aCol = t * TS + threadIdx.x;
        int bRow = t * TS + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.f;

        __syncthreads();   // ★ 等所有线程把 tile 装完

        // 在 shared 里做 TS 次乘加
        #pragma unroll
        for (int k = 0; k < TS; ++k) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        __syncthreads();   // ★ 等所有线程算完，再切下一个 tile
    }

    if (row < M && col < N) C[row * N + col] = sum;
}

int main() {
    const int M = 1024, N = 1024, K = 1024;
    const size_t bytesA = M * K * sizeof(float);
    const size_t bytesB = K * N * sizeof(float);
    const size_t bytesC = M * N * sizeof(float);

    float *h_A = (float*)std::malloc(bytesA);
    float *h_B = (float*)std::malloc(bytesB);
    float *h_C = (float*)std::malloc(bytesC);
    for (int i = 0; i < M * K; ++i) h_A[i] = 1.f;
    for (int i = 0; i < K * N; ++i) h_B[i] = 2.f;

    float *d_A, *d_B, *d_C;
    MUSA_CHECK(musaMalloc(&d_A, bytesA));
    MUSA_CHECK(musaMalloc(&d_B, bytesB));
    MUSA_CHECK(musaMalloc(&d_C, bytesC));
    MUSA_CHECK(musaMemcpy(d_A, h_A, bytesA, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, bytesB, musaMemcpyHostToDevice));

    dim3 block(TS, TS);
    dim3 grid((N + TS - 1) / TS, (M + TS - 1) / TS);

    musaEvent_t t0, t1;
    musaEventCreate(&t0); musaEventCreate(&t1);

    gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);   // warmup
    MUSA_CHECK(musaDeviceSynchronize());

    musaEventRecord(t0);
    const int REPEAT = 10;
    for (int r = 0; r < REPEAT; ++r) {
        gemm_tiled<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
    }
    musaEventRecord(t1);
    musaEventSynchronize(t1);

    float ms = 0.f; musaEventElapsedTime(&ms, t0, t1);
    ms /= REPEAT;
    double gflops = 2.0 * M * N * K / (ms * 1e6);

    MUSA_CHECK(musaMemcpy(h_C, d_C, bytesC, musaMemcpyDeviceToHost));
    printf("Tiled GEMM  M=N=K=%d  TS=%d  time=%.2f ms  perf=%.1f GFLOPS  C[0]=%.0f\n",
           M, TS, ms, gflops, h_C[0]);

    musaFree(d_A); musaFree(d_B); musaFree(d_C);
    std::free(h_A); std::free(h_B); std::free(h_C);
    musaEventDestroy(t0); musaEventDestroy(t1);
    return 0;
}


// ╔══════════════════════════════════════════════════════════════════════════╗
// ║  PART III  ─  练习题与解答（对应 exercises.md E3.1 / E3.2 / E3.5）         ║
// ╚══════════════════════════════════════════════════════════════════════════╝

// ★ Q1: TS 扫描（8 / 16 / 32 / 48）
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 改 TS 跑 4 组对比 GFLOPS。
//
//  ▸ 典型规律：
//      TS=8   → block 只有 64 线程，warp 利用率低，慢
//      TS=16  → 256 线程/block，sweet spot
//      TS=32  → 1024 线程/block，可能更快也可能更慢（看 occupancy）
//      TS=48  → 1152 线程/block 超过 maxThreadsPerBlock，启动失败
//
//  ▸ 关键：
//      TS² ≤ maxThreadsPerBlock（1024 → TS ≤ 32）
//      shared 用量 = 2 * TS² * 4 字节，不能超过 sharedMemPerBlock

// ★ Q2: 矩阵尺寸扫描（256/512/1024/2048/4096）
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 固定 TS=16，扫描 M=N=K。
//
//  ▸ 典型规律：
//      • 小矩阵（256）：grid 太小，没填满 SM，GFLOPS 低
//      • 中等（1024~2048）：进入稳定区
//      • 大矩阵（4096）：可能因 cache miss 增多略降
//
//  ▸ 启示：测性能要矩阵足够大，否则数据不可信。

// ★ Q3: Bank Conflict 验证
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 改写：__shared__ float tileA[TS][TS+1];   // 加 +1 padding
//          __shared__ float tileB[TS][TS+1];
//
//  ▸ 思路：
//      原来 tileA[k][threadIdx.x]：同 warp 32 个线程同步访问同一行不同列。
//      如果 TS=32，列方向恰好 32 个 bank → 完美无冲突；
//      但如果某些访问模式（如 tileB[k][..]） warp 内不同线程访问同 bank → 冲突。
//      加 padding 让 stride 错开。
//
//  ▸ 实测可能差距不大（取决于硬件、SDK），建议用 profiler 看具体冲突计数。

// ★ Q4 (扩展): 寄存器分块（Register Tiling）
// ──────────────────────────────────────────────────────────────────────────
//
//  ▸ 当前是"shared memory tiling"，每线程算 1 个 C 元素。
//    再进一步：让每线程算多个 C 元素（比如 4×4），用寄存器存中间结果。
//
//  ▸ 这是 cuBLAS / muBLAS 性能的核心，能再提速 5~10x。
//    实现细节复杂，超出本周范围，了解概念即可。
// ============================================================================
