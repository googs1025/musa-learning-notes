// ============================================================================
//  文件:63_interleave.mu
//  来源:LeetGPU Easy · #63 · Interleave Two Arrays
//        https://leetgpu.com/challenges/interleave-two-arrays
//  考点:输入索引 vs 输出索引不一致 / 输出 stride=2 的写入
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:A=[a0,a1,...], B=[b0,b1,...],各长 N,
//          输出 [a0, b0, a1, b1, ...](长度 2N)。
//
//  PART II  解答:每 thread 处理一对 (A[i], B[i]),
//          写到 output[2*i] 和 output[2*i+1]。
//
//  PART III  访存模式:
//          • A、B 的读是 coalesced(相邻 thread 读相邻地址)
//          • output 的写是 stride=2,即 thread i 写 2i,严格意义上不算 coalesced
//          • 但 MUSA/CUDA 的 L1/L2 缓存可以合并相邻写入的两个 cache line,
//            实测损失不大;真正成为瓶颈再考虑用 vector store

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void interleave_kernel(const float* A, const float* B,
                                  float* output, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        output[2 * i + 0] = A[i];
        output[2 * i + 1] = B[i];
    }
}

extern "C" void solve(const float* A, const float* B, float* output, int N) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    interleave_kernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, output, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test ───────────────────────────────────────────────────────
//  A=[1,2,3], B=[10,20,30]  →  [1,10,2,20,3,30]
int main() {
    const int N = 3;
    float h_A[N] = {1, 2, 3};
    float h_B[N] = {10, 20, 30};
    float h_out[2 * N] = {};

    float *d_A, *d_B, *d_out;
    MUSA_CHECK(musaMalloc(&d_A,   sizeof(h_A)));
    MUSA_CHECK(musaMalloc(&d_B,   sizeof(h_B)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_A, h_A, sizeof(h_A), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_B, h_B, sizeof(h_B), musaMemcpyHostToDevice));

    solve(d_A, d_B, d_out, N);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < 2 * N; ++i) std::printf(" %.1f", h_out[i]);
    std::printf("   (expect: 1.0 10.0 2.0 20.0 3.0 30.0)\n");

    musaFree(d_A); musaFree(d_B); musaFree(d_out);
    return 0;
}
