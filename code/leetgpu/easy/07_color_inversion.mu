// ============================================================================
//  文件:07_color_inversion.mu
//  来源:LeetGPU Easy · #7 · Color Inversion
//        https://leetgpu.com/challenges/color-inversion
//  考点:1D 索引 + uchar 数据 / 部分字段(RGB 反色,A 不动)
//  roadmap:Week 1(线程索引,练 stride 越界)
// ============================================================================
//
//  PART I  题面:image 是 RGBA 一维数组,长度 width*height*4。
//          每个像素 4 字节,R/G/B 各做 255-x,A 不变。
//
//  PART II  解答:每个 thread 处理一个"像素"(=4 字节),
//          需要注意:grid 维度按像素数算,kernel 内一次写 3 字节。
//
//  PART III  陷阱:starter 的 blocksPerGrid 用 width*height 而不是
//          width*height*4——题目就是按像素并行,不是按字节。

#include <musa_runtime.h>
#include "musa_common.h"
#include <cstdio>

__global__ void invert_kernel(unsigned char* image, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pixels = width * height;
    if (idx < total_pixels) {
        int base = idx * 4;
        image[base + 0] = 255 - image[base + 0];  // R
        image[base + 1] = 255 - image[base + 1];  // G
        image[base + 2] = 255 - image[base + 2];  // B
        // image[base + 3] (A) 保持不变
    }
}

extern "C" void solve(unsigned char* image, int width, int height) {
    int threadsPerBlock = 256;
    int blocksPerGrid   = (width * height + threadsPerBlock - 1) / threadsPerBlock;
    invert_kernel<<<blocksPerGrid, threadsPerBlock>>>(image, width, height);
    musaDeviceSynchronize();
}

// ── 本地 smoke test(Example 1)─────────────────────────────────────────────
//  [255,0,128,255,  0,255,0,255]  →  [0,255,127,255,  255,0,255,255]
int main() {
    const int width = 1, height = 2;
    const int N = width * height * 4;
    unsigned char h_img[N] = {255, 0, 128, 255,  0, 255, 0, 255};

    unsigned char* d_img;
    MUSA_CHECK(musaMalloc(&d_img, N));
    MUSA_CHECK(musaMemcpy(d_img, h_img, N, musaMemcpyHostToDevice));

    solve(d_img, width, height);

    MUSA_CHECK(musaMemcpy(h_img, d_img, N, musaMemcpyDeviceToHost));
    std::printf("img =");
    for (int i = 0; i < N; ++i) std::printf(" %d", h_img[i]);
    std::printf("   (expect: 0 255 127 255 255 0 255 255)\n");

    musaFree(d_img);
    return 0;
}
