// ============================================================================
//  文件:24_rainbow_table.mu
//  来源:LeetGPU Easy · #24 · Rainbow Table(FNV-1a hash × R 轮)
//        https://leetgpu.com/challenges/rainbow-table
//  考点:__device__ 函数 / 同一元素多轮迭代 / 寄存器内循环
//  roadmap:Week 1(用 device 函数模块化)
// ============================================================================
//
//  PART I  题面:对每个 int 用给定的 FNV-1a 哈希函数迭代 R 轮,
//          R 轮的输出 = 第 R 次 hash 之后的值。
//
//  PART II  解答:外层用 thread 平铺数据,内层每 thread 自己迭代 R 次。
//          中间结果可放在寄存器(unsigned int local),无需写回 global。
//
//  PART III  细节:
//          • 题目签名输入是 const int*,但哈希要按 unsigned 解释,转一下
//          • 中间循环不要回写 output[i] —— 每轮写 global 又读,带宽爆炸
//          • fnv1a_hash 是 __device__,可在 kernel 内调用,不能 host 调

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__device__ unsigned int fnv1a_hash(unsigned int input) {
    const unsigned int FNV_PRIME    = 16777619u;
    const unsigned int OFFSET_BASIS = 2166136261u;
    unsigned int hash = OFFSET_BASIS;
    for (int byte_pos = 0; byte_pos < 4; ++byte_pos) {
        unsigned char b = (input >> (byte_pos * 8)) & 0xFFu;
        hash = (hash ^ b) * FNV_PRIME;
    }
    return hash;
}

__global__ void fnv1a_hash_kernel(const int* input, unsigned int* output,
                                  int N, int R) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        unsigned int h = static_cast<unsigned int>(input[i]);
        for (int r = 0; r < R; ++r) h = fnv1a_hash(h);
        output[i] = h;
    }
}

extern "C" void solve(const int* input, unsigned int* output, int N, int R) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (N + threadsPerBlock - 1) / threadsPerBlock;
    fnv1a_hash_kernel<<<blocksPerGrid, threadsPerBlock>>>(input, output, N, R);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [123,456,789], R=2  →  [1636807824, 1273011621, 2193987222]
int main() {
    const int N = 3, R = 2;
    int h_in[N]            = {123, 456, 789};
    unsigned int h_out[N]  = {};

    int*           d_in;
    unsigned int*  d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, N, R);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < N; ++i) std::printf(" %u", h_out[i]);
    std::printf("   (expect: 1636807824 1273011621 2193987222)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
