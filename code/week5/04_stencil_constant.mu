#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

// 5 点 stencil 的权重很小、所有线程都读同一组常量。
// 放进 constant memory 可以让硬件广播，避免每个线程都从 global 重复取权重。
__constant__ float c_w[5];

__global__ void stencil5(const float* in, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 && i < n - 2) {
        float v = 0.0f;
        #pragma unroll
        for (int k = -2; k <= 2; ++k) v += in[i + k] * c_w[k + 2];
        out[i] = v;
    }
}

int main() {
    const int N = 1 << 20;
    float w[5] = {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f};
    float *in = nullptr, *out = nullptr;
    MUSA_CHECK(musaMalloc(&in, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&out, N * sizeof(float)));

    // Host -> constant symbol。符号名 c_w 是 device 端全局常量数组。
    MUSA_CHECK(musaMemcpyToSymbol(c_w, w, sizeof(w)));

    dim3 block(256), grid((N + 255) / 256);
    stencil5<<<grid, block>>>(in, out, N);
    MUSA_CHECK_KERNEL();
    std::printf("constant stencil done\n");

    MUSA_CHECK(musaFree(in));
    MUSA_CHECK(musaFree(out));
}
