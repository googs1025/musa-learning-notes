// ============================================================================
//  文件:65_geglu.mu
//  来源:LeetGPU Easy · #65 · GEGLU(GELU-gated GLU)
//        https://leetgpu.com/challenges/geglu
//  考点:erff / 跨半区索引 / GELU 的精确版而非近似版
//  roadmap:Week 2(LLM 算子家族)
// ============================================================================
//
//  PART I  题面:输入 N(偶数),前半 x1、后半 x2。
//          GEGLU(x1, x2) = x1 * GELU(x2),
//          GELU(x) = 0.5 * x * (1 + erf(x / sqrt(2)))。
//
//  PART II  解答:每 thread 算一个 i ∈ [0, halfN),取 x1=input[i],
//          x2=input[i+halfN],按公式写出。
//
//  PART III  GELU 的两个版本:
//          • 精确版:用 erff(本题就是这种)
//          • Tanh 近似版:GELU(x) ≈ 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
//            GPT-2 / BERT 早期常用近似;LLaMA 已经不再用 GELU 改用 SwiGLU。
//          • 1/√2 ≈ 0.70710678,这里直接写常数避免运行时算

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void geglu_kernel(const float* input, float* output, int halfN) {
    constexpr float INV_SQRT2 = 0.70710678118654752440f;  // 1/√2
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float gelu_x2 = 0.5f * x2 * (1.0f + erff(x2 * INV_SQRT2));
        output[i] = x1 * gelu_x2;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    geglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [1.0, 1.0]  →  GELU(1.0) ≈ 0.8413  (x1=1 * GELU(1))
int main() {
    const int N = 2, halfN = N / 2;
    float h_in[N]      = {1.0f, 1.0f};
    float h_out[halfN] = {};

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, N);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < halfN; ++i) std::printf(" %.4f", h_out[i]);
    std::printf("   (expect: 0.8413)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
