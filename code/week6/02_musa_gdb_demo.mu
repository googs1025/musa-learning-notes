// ============================================================================
//  文件：02_musa_gdb_demo.mu
//  标题：Week 6 · 示例 1 · 主动触发 illegal address
//  目标：练习异步错误定位、MUSA GDB、Error Dump 工作流
// ============================================================================
//
//  编译：make          运行：./02_musa_gdb_demo
//
//  注意：本程序在真实 MUSA 硬件上预期失败。它故意删除边界检查，让
//  kernel 写出已分配数组范围，用于观察错误码和调试工具输出。

#include <musa_runtime.h>
#include <cstdio>
#include <cstdlib>

#define MUSA_CHECK(call) do {                                       \
    musaError_t _e = (call);                                   \
    if (_e != musaSuccess) {                                   \
        fprintf(stderr, "MUSA error %d at %s:%d\n",            \
                (int)_e, __FILE__, __LINE__);                  \
        std::exit(1);                                          \
    }                                                          \
} while (0)

__global__ void write_oob(float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    out[i] = float(i);  // 故意缺少 if (i < n)
}

int main() {
    const int N = 1024;
    const int threads = 256;
    const int blocks = 8;  // launch 2048 threads, but only allocate 1024 floats
    const size_t bytes = N * sizeof(float);

    float* d_out = nullptr;
    MUSA_CHECK(musaMalloc(&d_out, bytes));

    printf("Launching intentionally broken kernel: allocated=%d floats, launched=%d threads\n",
           N, threads * blocks);

    write_oob<<<blocks, threads>>>(d_out, N);
    MUSA_CHECK(musaGetLastError());       // launch-configuration errors
    MUSA_CHECK(musaDeviceSynchronize());  // expected async illegal-address report

    MUSA_CHECK(musaFree(d_out));
    printf("Unexpected success. If this happens, increase blocks or check SDK diagnostics.\n");
    return 0;
}
