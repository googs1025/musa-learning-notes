// ============================================================================
//  文件:08_matrix_addition.mu
//  来源:LeetGPU Easy · #8 · Matrix Addition
//        https://leetgpu.com/challenges/matrix-addition
//  考点:把 2D 数据当 1D 处理 / N*N 不是 N(grid 大小要注意)
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:C[N×N] = A + B,逐元素相加,row-major float32。
//
//  PART II  解答:row-major 下,矩阵加法 == 长度 N*N 的一维向量加法。
//          这是把 2D 转 1D 处理的最简单例子。
//
//  PART III  Why 1D?
//          • 逐元素加法没有相邻数据复用 → 2D 网格没好处
//          • 一维线性化,coalesce 完美,launch 配置简单
//          • 后续 GEMM/transpose 才必须用 2D(因为有 2D 邻域访问)

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void matrix_add(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (i < total) C[i] = A[i] + B[i];
}

extern "C" void solve(const float* A, const float* B, float* C, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N * N + threadsPerBlock - 1) / threadsPerBlock;
    matrix_add<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [[1,2],[3,4]] + [[5,6],[7,8]] = [[6,8],[10,12]]
int main() {
    const int N = 2;
    float h_A[] = {1, 2, 3, 4};
    float h_B[] = {5, 6, 7, 8};
    float h_C[N * N] = {};

    float *d_A, *d_B, *d_C;
    MUSA_CHECK(musaMalloc(&d_A, sizeof(h_A)));
    MUSA_CHECK(musaMalloc(&d_B, sizeof(h_B)));
    MUSA_CHECK(musaMalloc(&d_C, sizeof(h_C)));
    MUSA_CHECK(musaMemcpy(d_A, h_A, sizeof(h_A), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, sizeof(h_B), musaMemcpyHostToDevice));

    solve(d_A, d_B, d_C, N);

    MUSA_CHECK(musaMemcpy(h_C, d_C, sizeof(h_C), musaMemcpyDeviceToHost));
    std::printf("C =");
    for (int i = 0; i < N * N; ++i) std::printf(" %.1f", h_C[i]);
    std::printf("   (expect: 6.0 8.0 10.0 12.0)\n");

    musaFree(d_A); musaFree(d_B); musaFree(d_C);
    return 0;
}
