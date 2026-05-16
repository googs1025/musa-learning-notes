// ============================================================================
//  文件:01_vector_add.mu
//  来源:LeetGPU Easy · #1 · Vector Addition
//        https://leetgpu.com/challenges/vector-addition
//  考点:1D grid 边界 / 元素级并行最基础模板
//  roadmap:Week 1(线程索引 + 显存四件套)
// ============================================================================
//
//  PART I  题目(中文摘录)
//  ──────────────────────────────────────────────────────────────────────────
//  实现 C = A + B,逐元素相加,float32,长度 N。
//  约束:1 ≤ N ≤ 1e8,性能基准 N = 25,000,000。
//
//  PART II  解答(MUSA kernel + solve)
//  ──────────────────────────────────────────────────────────────────────────
//  ★ 提交 LeetGPU 时把 musa 换成 cuda 即可,solve 签名完全一致。
//
//  PART III  MUSA vs CUDA 移植差异
//  ──────────────────────────────────────────────────────────────────────────
//  • <cuda_runtime.h>   → <musa_runtime.h>
//  • cudaDeviceSynchronize → musaDeviceSynchronize
//  • kernel 体本身 0 改动(没用 warp-level intrinsics)
//  • MUSA warp = 128(CUDA 32),但本题只跑 elementwise,不受影响

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

// ── 解答 kernel ─────────────────────────────────────────────────────────────
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

// LeetGPU 入口:device 指针由 judge 准备好
extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
int main() {
    const int N = 4;
    float h_A[N] = {1.0f, 2.0f, 3.0f, 4.0f};
    float h_B[N] = {5.0f, 6.0f, 7.0f, 8.0f};
    float h_C[N] = {};

    float *d_A, *d_B, *d_C;
    MUSA_CHECK(musaMalloc(&d_A, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&d_B, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&d_C, N * sizeof(float)));
    MUSA_CHECK(musaMemcpy(d_A, h_A, N * sizeof(float), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, N * sizeof(float), musaMemcpyHostToDevice));

    solve(d_A, d_B, d_C, N);

    MUSA_CHECK(musaMemcpy(h_C, d_C, N * sizeof(float), musaMemcpyDeviceToHost));
    std::printf("C =");
    for (int i = 0; i < N; ++i) std::printf(" %.1f", h_C[i]);
    std::printf("   (expect: 6.0 8.0 10.0 12.0)\n");

    musaFree(d_A); musaFree(d_B); musaFree(d_C);
    return 0;
}
