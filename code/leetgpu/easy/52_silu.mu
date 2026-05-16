// ============================================================================
//  文件:52_silu.mu
//  来源:LeetGPU Easy · #52 · SiLU(Sigmoid Linear Unit / Swish)
//        https://leetgpu.com/challenges/silu
//  考点:1D 元素 / expf / 复合数学函数
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:SiLU(x) = x · σ(x),σ(x) = 1 / (1 + e^-x)。
//          常被叫 Swish(Google)。LLaMA / Qwen 等模型常用 SwiGLU 的核心激活。
//
//  PART II  解答:一次 expf + 两次乘除。
//
//  PART III  数值稳定:
//          • x 很负时 e^-x 会很大 → 1/(1+e^-x) ≈ 0,无溢出,安全
//          • x 很正时 e^-x ≈ 0,σ≈1,也安全
//          • 不需要像 softmax 那样减最大值;easy 题就是把公式搬上 GPU
//          • 想再快可以用 __expf(快但精度低),easy 阶段不必

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void silu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float x = input[i];
        float sigmoid = 1.0f / (1.0f + expf(-x));
        output[i] = x * sigmoid;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    silu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [0.5, 1.0, -0.5]  →  [0.3112, 0.7311, -0.1888]
int main() {
    const int N = 3;
    float h_in[N]  = {0.5f, 1.0f, -0.5f};
    float h_out[N] = {};

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, N);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < N; ++i) std::printf(" %.4f", h_out[i]);
    std::printf("   (expect: 0.3112 0.7311 -0.1888)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
