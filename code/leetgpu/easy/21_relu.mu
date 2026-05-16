// ============================================================================
//  文件:21_relu.mu
//  来源:LeetGPU Easy · #21 · ReLU
//        https://leetgpu.com/challenges/relu
//  考点:1D 元素 / 最简激活函数 / fmaxf
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:output[i] = max(0, input[i])。
//
//  PART II  解答:用 fmaxf 比 if/else 短一行,也能避免分支(编译器会用 mufmaxf)。
//
//  PART III  小知识:
//          • fmaxf 在 MUSA/CUDA 都是 device 内置函数,对应硬件 max 指令
//          • 不要写 input[i] > 0 ? input[i] : 0 —— 看似一样,编译器有时
//            会保留分支(虽然 SIMT 上分支 cost 取决于 warp 内是否分歧)

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void relu_kernel(const float* input, float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) output[i] = fmaxf(0.0f, input[i]);
}

extern "C" void solve(const float* input, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    relu_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [-2,-1,0,1,2]  →  [0,0,0,1,2]
int main() {
    const int N = 5;
    float h_in[N]  = {-2, -1, 0, 1, 2};
    float h_out[N] = {};

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, N);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < N; ++i) std::printf(" %.1f", h_out[i]);
    std::printf("   (expect: 0.0 0.0 0.0 1.0 2.0)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
