// ============================================================================
//  文件:03_matrix_transpose.mu
//  来源:LeetGPU Easy · #3 · Matrix Transpose
//        https://leetgpu.com/challenges/matrix-transpose
//  考点:2D 线程索引 / 读写不对称的 coalesce 直觉
//  roadmap:Week 3(显存层级) / Week 4(coalesced access)
// ============================================================================
//
//  PART I  题面:输入 input[rows × cols],输出 output[cols × rows],
//          满足 output[j][i] = input[i][j],row-major。
//
//  PART II  解答:每个 thread 处理一个 (row, col) 元素。
//
//  PART III  优化空间(留给 Week 4):
//          • naive 版本的 output 写入是 strided(非合并)→ 带宽差
//          • 经典优化:用 shared memory tile + padding 避免 bank conflict
//            (CUDA 上叫 transpose-tiled-no-bank-conflict,MUSA 同思路)
//          • 本文件先给 naive 基线,后续在 Week 4 给出 tiled 版本对比

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void matrix_transpose_kernel(const float* input, float* output,
                                        int rows, int cols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // 0 .. cols-1
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // 0 .. rows-1
    if (row < rows && col < cols) {
        output[col * rows + row] = input[row * cols + col];
    }
}

extern "C" void solve(const float* input, float* output, int rows, int cols) {
    dim3 block(16, 16);
    dim3 grid((cols + block.x - 1) / block.x,
              (rows + block.y - 1) / block.y);
    matrix_transpose_kernel<<<grid, block>>>(input, output, rows, cols);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [[1,2,3],[4,5,6]]^T = [[1,4],[2,5],[3,6]]
int main() {
    const int rows = 2, cols = 3;
    float h_in[]  = {1, 2, 3, 4, 5, 6};
    float h_out[rows * cols] = {};

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, rows, cols);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < rows * cols; ++i) std::printf(" %.1f", h_out[i]);
    std::printf("   (expect: 1.0 4.0 2.0 5.0 3.0 6.0)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
