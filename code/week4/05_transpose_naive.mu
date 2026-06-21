#include "musa_common.h"
#include <cstdio>

// 朴素转置：读 in[y][x] 是连续的，但写 out[x][y] 是跨步的。
// 这个例子作为 Week4 shared transpose 的 baseline。
__global__ void transpose_naive(const float* in, float* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < w && y < h) out[x * h + y] = in[y * w + x];
}

int main() {
    const int W = 2048, H = 2048;
    float *in = nullptr, *out = nullptr;
    MUSA_CHECK(musaMalloc(&in, W * H * sizeof(float)));
    MUSA_CHECK(musaMalloc(&out, W * H * sizeof(float)));

    dim3 block(16, 16);
    dim3 grid((W + 15) / 16, (H + 15) / 16);
    transpose_naive<<<grid, block>>>(in, out, W, H);
    MUSA_CHECK_KERNEL();

    GpuTimer t;
    t.start();
    for (int r = 0; r < 20; ++r) transpose_naive<<<grid, block>>>(in, out, W, H);
    t.stop();
    MUSA_CHECK(musaDeviceSynchronize());
    std::printf("transpose naive %.3f ms\n", t.elapsed_ms() / 20.0f);

    MUSA_CHECK(musaFree(in));
    MUSA_CHECK(musaFree(out));
}
