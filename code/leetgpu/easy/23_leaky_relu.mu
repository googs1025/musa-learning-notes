// ============================================================================
//  文件:23_leaky_relu.mu
//  来源:LeetGPU Easy · #23 · Leaky ReLU(α=0.01)
//        https://leetgpu.com/challenges/leaky-relu
//  考点:1D 元素 / 分段函数
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:x>0 时 y=x,否则 y=αx,α=0.01。
//
//  PART II  解答:三目表达式即可。也可用 (x > 0 ? 1.0f : alpha) * x。
//
//  PART III  分支 vs 乘法:
//          • SIMT 下同 warp 内取值不同时会产生分歧(divergence)
//          • 但 leaky_relu 的两路计算量都很轻,几乎看不出差
//          • 也有用 (x + abs(x))/2 + alpha*(x - abs(x))/2 的无分支写法,
//            实测对编译器优化后差距不明显——可读性优先,用三目就好。

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void leaky_relu_kernel(const float* input, float* output, int N) {
    constexpr float ALPHA = 0.01f;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        output[i] = (x > 0.0f) ? x : ALPHA * x;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    leaky_relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [1,-2,3,-4]  →  [1, -0.02, 3, -0.04]
int main() {
    const int N = 4;
    float h_in[N]  = {1, -2, 3, -4};
    float h_out[N] = {};

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, N);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < N; ++i) std::printf(" %.3f", h_out[i]);
    std::printf("   (expect: 1.000 -0.020 3.000 -0.040)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
