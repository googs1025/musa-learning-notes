// ============================================================================
//  文件:62_value_clipping.mu
//  来源:LeetGPU Easy · #62 · Value Clipping(torch.clamp)
//        https://leetgpu.com/challenges/value-clipping
//  考点:1D 元素 / fminf+fmaxf 复合 / 标量参数传值
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:y = clamp(x, lo, hi),把 x 限制在 [lo, hi] 区间。
//
//  PART II  解答:fminf(fmaxf(x, lo), hi)。
//
//  PART III  小知识:
//          • lo/hi 是 host 标量,传给 kernel 时按值拷贝(走 register),
//            不需要走 device memory
//          • 看似平凡的 clamp 是量化(quantization)和激活稳定的常用预处理

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void clip_kernel(const float* input, float* output,
                            float lo, float hi, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[i] = fminf(fmaxf(input[i], lo), hi);
    }
}

extern "C" void solve(const float* input, float* output,
                      float lo, float hi, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    clip_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, lo, hi, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [1.5,-2.0,3.0,4.5], lo=0.0, hi=3.5  →  [1.5,0.0,3.0,3.5]
int main() {
    const int N = 4;
    float h_in[N]  = {1.5f, -2.0f, 3.0f, 4.5f};
    float h_out[N] = {};
    float lo = 0.0f, hi = 3.5f;

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, lo, hi, N);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < N; ++i) std::printf(" %.1f", h_out[i]);
    std::printf("   (expect: 1.5 0.0 3.0 3.5)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
