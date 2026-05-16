// ============================================================================
//  文件:19_reverse_array.mu
//  来源:LeetGPU Easy · #19 · Reverse Array(in-place)
//        https://leetgpu.com/challenges/reverse-array
//  考点:in-place 写回 / 配对线程 / 避免重复 swap
//  roadmap:Week 1(线程索引,经典 "pair access" 模式)
// ============================================================================
//
//  PART I  题面:in-place 反转长度 N 的 float 数组。
//
//  PART II  解答:让 thread i 同时持有 input[i] 和 input[N-1-i],
//          一次 swap 完成两端互换。
//          ★ 关键:只让 i < N/2 的线程动手,否则 swap 会被做两次回到原样。
//
//  PART III  替代写法:
//          • 也可以让每 thread 只写自己位置(读 N-1-i),需要"读全局后做 barrier
//            再写"——但 grid 间没有 barrier,只能用辅助数组,反而麻烦。
//          • in-place + 配对 swap 是最简单的并行写法。

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void reverse_array(float* input, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int half = N / 2;                   // 只让前一半线程做事
    if (i < half) {
        int j = N - 1 - i;
        float tmp  = input[i];
        input[i]   = input[j];
        input[j]   = tmp;
    }
}

extern "C" void solve(float* input, int N) {
    int threadsPerBlock = 256;
    // 只需要覆盖前一半即可;为了和 starter 保持一致,这里仍按 N 算
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    reverse_array<<<blocksPerGrid, threadsPerBlock>>>(input, N);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [1,2,3,4]  →  [4,3,2,1]
int main() {
    const int N = 4;
    float h_arr[N] = {1, 2, 3, 4};

    float* d_arr;
    MUSA_CHECK(musaMalloc(&d_arr, sizeof(h_arr)));
    MUSA_CHECK(musaMemcpy(d_arr, h_arr, sizeof(h_arr), musaMemcpyHostToDevice));

    solve(d_arr, N);

    MUSA_CHECK(musaMemcpy(h_arr, d_arr, sizeof(h_arr), musaMemcpyDeviceToHost));
    std::printf("arr =");
    for (int i = 0; i < N; ++i) std::printf(" %.1f", h_arr[i]);
    std::printf("   (expect: 4.0 3.0 2.0 1.0)\n");

    musaFree(d_arr);
    return 0;
}
