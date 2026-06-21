#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

__global__ void matrix_to_row_sums(const float* m, float* rows, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x < width && y < height) atomicAdd(&rows[y], m[y * width + x]);
}

int main() {
    const int W = 1024, H = 1024;
    float* h = (float*)std::malloc(W * H * sizeof(float));
    float* rows = (float*)std::calloc(H, sizeof(float));
    for (int i = 0; i < W * H; ++i) h[i] = 1.0f;
    float *d = nullptr, *r = nullptr;
    MUSA_CHECK(musaMalloc(&d, W * H * sizeof(float)));
    MUSA_CHECK(musaMalloc(&r, H * sizeof(float)));
    MUSA_CHECK(musaMemcpy(d, h, W * H * sizeof(float), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemset(r, 0, H * sizeof(float)));
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    matrix_to_row_sums<<<grid, block>>>(d, r, W, H);
    MUSA_CHECK_KERNEL();
    MUSA_CHECK(musaMemcpy(rows, r, H * sizeof(float), musaMemcpyDeviceToHost));
    std::printf("row0=%.0f expected=%d row_last=%.0f\n", rows[0], W, rows[H-1]);
    MUSA_CHECK(musaFree(d)); MUSA_CHECK(musaFree(r)); std::free(h); std::free(rows);
}
