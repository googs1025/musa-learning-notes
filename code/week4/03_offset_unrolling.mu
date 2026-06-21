#include "musa_common.h"
#include <cstdio>

__global__ void offset_unroll4(const float* in, float* out, int n, int off) {
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 4;

    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        if (i + k < n) out[i + k] = in[i + k + off];
    }
}

int main() {
    const int N = 1 << 24;
    float *in = nullptr, *out = nullptr;
    MUSA_CHECK(musaMalloc(&in, (N + 64) * sizeof(float)));
    MUSA_CHECK(musaMalloc(&out, N * sizeof(float)));

    dim3 block(256);
    dim3 grid((N + 1023) / 1024);
    for (int off : {0, 1, 8, 31}) {
        offset_unroll4<<<grid, block>>>(in, out, N, off);
        MUSA_CHECK_KERNEL();

        GpuTimer t;
        t.start();
        for (int r = 0; r < 50; ++r) {
            offset_unroll4<<<grid, block>>>(in, out, N, off);
        }
        t.stop();
        MUSA_CHECK(musaDeviceSynchronize());
        std::printf("offset=%2d unroll4=%.3f ms\n", off, t.elapsed_ms() / 50.0f);
    }

    MUSA_CHECK(musaFree(in));
    MUSA_CHECK(musaFree(out));
}
