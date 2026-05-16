// ============================================================================
//  文件:31_matrix_copy.mu
//  来源:LeetGPU Easy · #31 · Matrix Copy
//        https://leetgpu.com/challenges/matrix-copy
//  考点:1D 索引 / N*N 总长 / 退化成"显存拷贝"
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:B[i][j] = A[i][j],N×N,row-major float32。
//
//  PART II  解答:row-major 下整个矩阵就是连续 N*N 个 float,
//          一维 thread 平铺即可。
//
//  PART III  深思:
//          • 这其实就是 musaMemcpyDeviceToDevice 的功能,
//            真正生产代码一行 musaMemcpy 就完事。
//          • 题目用 kernel 写是为了练习"显存读写带宽天花板"——
//            一个完美 coalesced 的 copy kernel 通常是带宽测试基线。

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void copy_matrix_kernel(const float* A, float* B, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * N;
    if (i < total) B[i] = A[i];
}

extern "C" void solve(const float* A, float* B, int N) {
    int total = N * N;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (total + threadsPerBlock - 1) / threadsPerBlock;
    copy_matrix_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [[1,2],[3,4]] 原样拷到 B
int main() {
    const int N = 2;
    float h_A[] = {1, 2, 3, 4};
    float h_B[N * N] = {};

    float *d_A, *d_B;
    MUSA_CHECK(musaMalloc(&d_A, sizeof(h_A)));
    MUSA_CHECK(musaMalloc(&d_B, sizeof(h_B)));
    MUSA_CHECK(musaMemcpy(d_A, h_A, sizeof(h_A), musaMemcpyHostToDevice));

    solve(d_A, d_B, N);

    MUSA_CHECK(musaMemcpy(h_B, d_B, sizeof(h_B), musaMemcpyDeviceToHost));
    std::printf("B =");
    for (int i = 0; i < N * N; ++i) std::printf(" %.1f", h_B[i]);
    std::printf("   (expect: 1.0 2.0 3.0 4.0)\n");

    musaFree(d_A); musaFree(d_B);
    return 0;
}
