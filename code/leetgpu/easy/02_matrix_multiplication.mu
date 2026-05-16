// ============================================================================
//  文件:02_matrix_multiplication.mu
//  来源:LeetGPU Easy · #2 · Matrix Multiplication
//        https://leetgpu.com/challenges/matrix-multiplication
//  考点:2D 线程索引 / naive GEMM(为后续 tiled GEMM 做对照)
//  roadmap:Week 3(GEMM + 显存层级)
// ============================================================================
//
//  PART I  题面:C[M×K] = A[M×N] × B[N×K],row-major float32。
//          性能基准 M = N = K(大矩阵)。
//
//  PART II  解答:naive GEMM。每个 thread 算一个 C[row][col],
//           内层循环 N 次累加 A 的一行 · B 的一列。
//
//  PART III  MUSA 移植:仅头文件 + sync 改名,kernel 体不变。
//            naive 版受限于全局显存反复访问,Week 4 会用 shared memory tile
//            把 A、B 各搬一次到 SRAM,带宽利用率能上一个量级。

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void matrix_multiplication_kernel(const float* A, const float* B,
                                             float* C, int M, int N, int K) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 横坐标:K 方向
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 纵坐标:M 方向
    if (row < M && col < K) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
            acc += A[row * N + n] * B[n * K + col];
        }
        C[row * K + col] = acc;
    }
}

extern "C" void solve(const float* A, const float* B, float* C,
                      int M, int N, int K) {
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x,
              (M + block.y - 1) / block.y);
    matrix_multiplication_kernel<<<grid, block>>>(A, B, C, M, N, K);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
int main() {
    const int M = 2, N = 2, K = 2;
    float h_A[] = {1, 2, 3, 4};
    float h_B[] = {5, 6, 7, 8};
    float h_C[M * K] = {};

    float *d_A, *d_B, *d_C;
    MUSA_CHECK(musaMalloc(&d_A, M * N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&d_B, N * K * sizeof(float)));
    MUSA_CHECK(musaMalloc(&d_C, M * K * sizeof(float)));
    MUSA_CHECK(musaMemcpy(d_A, h_A, sizeof(h_A), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, sizeof(h_B), musaMemcpyHostToDevice));

    solve(d_A, d_B, d_C, M, N, K);

    MUSA_CHECK(musaMemcpy(h_C, d_C, sizeof(h_C), musaMemcpyDeviceToHost));
    std::printf("C =");
    for (int i = 0; i < M * K; ++i) std::printf(" %.1f", h_C[i]);
    std::printf("   (expect: 19.0 22.0 43.0 50.0)\n");

    musaFree(d_A); musaFree(d_B); musaFree(d_C);
    return 0;
}
