#include "musa_common.h"
#include <cstdio>

// 静态 shared memory：大小在编译期固定，适合 tile size 固定的 kernel。
__global__ void static_shared(float* out) {
    __shared__ float s[256];
    int t = threadIdx.x;

    s[t] = float(t);
    __syncthreads();  // 等整个 block 写完 shared 后再读，避免读到未初始化值。
    out[t] = s[255 - t];
}

// 动态 shared memory：大小在 launch 的第三个参数里传入。
// 适合 tile size 或临时 buffer 大小运行时才确定的场景。
__global__ void dynamic_shared(float* out) {
    extern __shared__ float s[];
    int t = threadIdx.x;

    s[t] = float(t * 2);
    __syncthreads();
    out[t] += s[t];
}

int main() {
    float* d = nullptr;
    MUSA_CHECK(musaMalloc(&d, 256 * sizeof(float)));

    static_shared<<<1, 256>>>(d);
    MUSA_CHECK_KERNEL();

    dynamic_shared<<<1, 256, 256 * sizeof(float)>>>(d);
    MUSA_CHECK_KERNEL();

    std::printf("shared basics done\n");
    MUSA_CHECK(musaFree(d));
}
