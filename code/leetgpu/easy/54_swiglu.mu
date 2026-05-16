// ============================================================================
//  文件:54_swiglu.mu
//  来源:LeetGPU Easy · #54 · SwiGLU
//        https://leetgpu.com/challenges/swish-gated-linear-unit
//  考点:把输入按维度切两半 / 不对称索引
//  roadmap:Week 2(为 LLM 推理算子打基础)
// ============================================================================
//
//  PART I  题面:输入长度 N(偶数),前一半 x1、后一半 x2。
//          output[i] = SiLU(x1[i]) * x2[i],输出长度 = N/2。
//          这是 LLaMA / Qwen 等模型 FFN 中的核心激活组合。
//
//  PART II  解答:每 thread 处理一个 halfN 内的位置,
//          x1 索引 = i,x2 索引 = i + halfN。
//
//  PART III  生产环境的常见变体:
//          • 真正的 SwiGLU 在 LLM 里是 gate_proj/up_proj/down_proj 三个 Linear,
//            这里只是其中"激活+逐元素乘"那一步
//          • 现实里这步常和 GEMM fuse 在一起(避免一次显存往返)

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void swiglu_kernel(const float* input, float* output, int halfN) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < halfN) {
        float x1 = input[i];
        float x2 = input[i + halfN];
        float silu = x1 / (1.0f + expf(-x1));   // x1 * sigmoid(x1)
        output[i] = silu * x2;
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int halfN = N / 2;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (halfN + threadsPerBlock - 1) / threadsPerBlock;
    swiglu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, halfN);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [1,2,3,4]  →  SiLU(1)*3, SiLU(2)*4 = 2.193..., 7.046...
int main() {
    const int N = 4, halfN = N / 2;
    float h_in[N]      = {1, 2, 3, 4};
    float h_out[halfN] = {};

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, N);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < halfN; ++i) std::printf(" %.4f", h_out[i]);
    std::printf("   (expect: 2.1932 7.0464)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
