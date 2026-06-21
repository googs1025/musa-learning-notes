#include "musa_common.h"
#include <cstdio>

// AoS(Array of Structs)：一个对象的字段挨在一起。
// 如果 kernel 只读 x/y，z/w 会占用同一个 cache line 但没有参与计算。
struct Particle{float x,y,z,w;};

__global__ void aos(const Particle* p, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = p[i].x + p[i].y;
}

// SoA(Struct of Arrays)：同一字段连续存放，更容易形成合并访存。
__global__ void soa(const float* x, const float* y, float* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = x[i] + y[i];
}

int main() {
    const int N = 1 << 24;
    Particle* p = nullptr;
    float *x = nullptr, *y = nullptr, *out = nullptr;
    MUSA_CHECK(musaMalloc(&p, N * sizeof(Particle)));
    MUSA_CHECK(musaMalloc(&x, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&y, N * sizeof(float)));
    MUSA_CHECK(musaMalloc(&out, N * sizeof(float)));

    dim3 block(256), grid((N + 255) / 256);

    GpuTimer t;
    aos<<<grid, block>>>(p, out, N);
    MUSA_CHECK_KERNEL();
    t.start();
    for (int r = 0; r < 30; ++r) aos<<<grid, block>>>(p, out, N);
    t.stop();
    float ta = t.elapsed_ms() / 30.0f;

    soa<<<grid, block>>>(x, y, out, N);
    MUSA_CHECK_KERNEL();
    t.start();
    for (int r = 0; r < 30; ++r) soa<<<grid, block>>>(x, y, out, N);
    t.stop();
    float ts = t.elapsed_ms() / 30.0f;

    std::printf("AoS=%.3f ms SoA=%.3f ms speedup=%.2fx\n", ta, ts, ta / ts);
    MUSA_CHECK(musaFree(p));
    MUSA_CHECK(musaFree(x));
    MUSA_CHECK(musaFree(y));
    MUSA_CHECK(musaFree(out));
}
