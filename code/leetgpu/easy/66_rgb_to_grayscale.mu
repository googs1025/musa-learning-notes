// ============================================================================
//  文件:66_rgb_to_grayscale.mu
//  来源:LeetGPU Easy · #66 · RGB → Grayscale
//        https://leetgpu.com/challenges/rgb-to-grayscale
//  考点:输入 stride=3,输出 stride=1 / 按像素并行
//  roadmap:Week 1
// ============================================================================
//
//  PART I  题面:gray = 0.299*R + 0.587*G + 0.114*B(ITU-R BT.601 标准系数)。
//          input 长度 = width*height*3(RGBRGB...),
//          output 长度 = width*height。
//
//  PART II  解答:每 thread 处理一个像素,基址 = idx*3 读 R/G/B 三个 float。
//
//  PART III  访存:
//          • 输入读 3 个 float / 像素,看似 stride=3 不 coalesced;
//            但 32 个 thread 连续 96 字节也是连续访问,被合并成 3 个 cache line,
//            带宽利用率仍 ok。
//          • 想极致优化可用 float3 / 向量化读,easy 阶段不必。

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output,
                                        int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx < total) {
        int base = idx * 3;
        float r = input[base + 0];
        float g = input[base + 1];
        float b = input[base + 2];
        output[idx] = 0.299f * r + 0.587f * g + 0.114f * b;
    }
}

extern "C" void solve(const float* input, float* output, int width, int height) {
    int total = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid   = (total + threadsPerBlock - 1) / threadsPerBlock;
    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(
        input, output, width, height);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  4 像素 RGB → 4 个灰度值
int main() {
    const int width = 2, height = 2;
    const int n = width * height;
    float h_in[n * 3] = {255.0f,   0.0f,   0.0f,
                           0.0f, 255.0f,   0.0f,
                           0.0f,   0.0f, 255.0f,
                         128.0f, 128.0f, 128.0f};
    float h_out[n] = {};

    float *d_in, *d_out;
    MUSA_CHECK(musaMalloc(&d_in,  sizeof(h_in)));
    MUSA_CHECK(musaMalloc(&d_out, sizeof(h_out)));
    MUSA_CHECK(musaMemcpy(d_in, h_in, sizeof(h_in), musaMemcpyHostToDevice));

    solve(d_in, d_out, width, height);

    MUSA_CHECK(musaMemcpy(h_out, d_out, sizeof(h_out), musaMemcpyDeviceToHost));
    std::printf("out =");
    for (int i = 0; i < n; ++i) std::printf(" %.3f", h_out[i]);
    std::printf("   (expect: 76.245 149.685 29.070 128.000)\n");

    musaFree(d_in); musaFree(d_out);
    return 0;
}
