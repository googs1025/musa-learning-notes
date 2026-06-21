#include "musa_common.h"
#include <cstdio>

__global__ void branch_kernel(float* out, int n, int mode) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float x = float(i & 255);
    if (mode == 0) {
        if (blockIdx.x & 1) x = x * 1.0001f + 1.0f;
        else x = x * 1.0001f + 1.0f;
    } else {
        if (threadIdx.x & 1) x = x * 1.0001f + 1.0f;
        else x = x / 1.0001f - 1.0f;
    }
    out[i] = x;
}

float run(float* d, int n, int mode) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);
    GpuTimer t;
    branch_kernel<<<grid, block>>>(d, n, mode);
    MUSA_CHECK_KERNEL();
    t.start();
    for (int r = 0; r < 50; ++r) branch_kernel<<<grid, block>>>(d, n, mode);
    t.stop();
    return t.elapsed_ms() / 50.0f;
}

int main() {
    const int N = 1 << 24;
    float* d = nullptr;
    MUSA_CHECK(musaMalloc(&d, N * sizeof(float)));
    float coherent = run(d, N, 0);
    float divergent = run(d, N, 1);
    std::printf("coherent=%.3f ms divergent=%.3f ms slowdown=%.2fx\n", coherent, divergent, divergent / coherent);
    MUSA_CHECK(musaFree(d));
}
