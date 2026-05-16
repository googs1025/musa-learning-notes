// ============================================================================
//  文件:09_1d_convolution.mu
//  来源:LeetGPU Easy · #9 · 1D Convolution(valid 边界)
//        https://leetgpu.com/challenges/1d-convolution
//  考点:窗口滑动 / 邻域重叠 → 为 shared memory tile 做铺垫
//  roadmap:Week 4(性能优化,shared mem 经典案例)
// ============================================================================
//
//  PART I  题面:output[i] = Σ_{j} input[i+j] * kernel[j],i ∈ [0, in-k]
//          输出长度 = input_size - kernel_size + 1。
//          注意题目说的是 "convolution",但定义其实是 cross-correlation
//          (kernel 不翻转)。深度学习里大多数库也这么用。
//
//  PART II  解答(naive):每 thread 算一个 output[i],内层 O(kernel_size)。
//          相邻 thread 读 input 区间重叠 → 实际带宽浪费严重。
//
//  PART III  优化(留给 Week 4):
//          • 把 input 的一块 tile 搬到 shared memory
//          • 多个 thread 复用 tile,把全局读 O(in*k) 降到 O(in)
//          • kernel_size 较大时(本题最大 2047)效果尤其明显

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void convolution_1d_kernel(const float* input, const float* kernel,
                                      float* output, int input_size, int kernel_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int output_size = input_size - kernel_size + 1;
    if (i < output_size) {
        float acc = 0.0f;
        for (int j = 0; j < kernel_size; ++j) {
            acc += input[i + j] * kernel[j];
        }
        output[i] = acc;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output,
                      int input_size, int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (output_size + threadsPerBlock - 1) / threadsPerBlock;
    convolution_1d_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, kernel, output, input_size, kernel_size);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  input=[1,2,3,4,5], kernel=[1,0,-1]  →  output=[-2,-2,-2]
int main() {
    float h_in[]  = {1, 2, 3, 4, 5};
    float h_k []  = {1, 0, -1};
    const int in_size = 5, k_size = 3;
    const int out_size = in_size - k_size + 1;
    float h_out[out_size] = {};

    float *d_in, *d_k, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_k,   sizeof(h_k)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_k,  h_k,  sizeof(h_k),  musaMemcpyHostToDevice));

    solve(d_in, d_k, d_out, in_size, k_size);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < out_size; ++i) std::printf(" %.1f", h_out[i]);
    std::printf("   (expect: -2.0 -2.0 -2.0)\n");

    musaFree(d_in); musaFree(d_k); musaFree(d_out);
    return 0;
}
