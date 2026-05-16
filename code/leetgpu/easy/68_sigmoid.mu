// ============================================================================
//  文件:68_sigmoid.mu
//  来源:LeetGPU Easy · #68 · Sigmoid
//        https://leetgpu.com/challenges/sigmoid
//  考点:1D 元素 / expf / 最经典激活函数
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:y = σ(x) = 1 / (1 + e^-x)。
//
//  PART II  解答:一次 expf 一次除法。
//
//  PART III  与 SiLU 的关系:
//          • SiLU(x) = x * Sigmoid(x)
//          • LLaMA 等模型用 SwiGLU,本质就是 Sigmoid 的复合应用
//          • 学到这一题已经离 transformer FFN 不远了

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void sigmoid_kernel(const float* X, float* Y, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        Y[i] = 1.0f / (1.0f + expf(-X[i]));
    }
}

extern "C" void solve(const float* X, float* Y, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    sigmoid_kernel<<<blocksPerGrid, threadsPerBlock>>>(X, Y, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [0,1,-1,2]  →  [0.5, 0.7311, 0.2689, 0.8808]
int main() {
    const int N = 4;
    float h_X[N] = {0, 1, -1, 2};
    float h_Y[N] = {};

    float *d_X, *d_Y;
    MUSA_CHECK(musaMalloc(&d_X, sizeof(h_X)));
    MUSA_CHECK(musaMalloc(&d_Y, sizeof(h_Y)));
    MUSA_CHECK(musaMemcpy(d_X, h_X, sizeof(h_X), musaMemcpyHostToDevice));

    solve(d_X, d_Y, N);

    MUSA_CHECK(musaMemcpy(h_Y, d_Y, sizeof(h_Y), musaMemcpyDeviceToHost));
    std::printf("Y =");
    for (int i = 0; i < N; ++i) std::printf(" %.4f", h_Y[i]);
    std::printf("   (expect: 0.5000 0.7311 0.2689 0.8808)\n");

    musaFree(d_X); musaFree(d_Y);
    return 0;
}
