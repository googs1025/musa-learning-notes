// ============================================================================
//  Case 01: Row-wise Softmax
//
//  目标:
//    给定 X[M, N], 每个 block 负责一行, 计算稳定 softmax。
//
//  学习点:
//    1. softmax 不能直接 exp(x), 要先减 row max 防止溢出。
//    2. 一行通常由多个线程分段扫描, 再做 block reduction。
//    3. reduction 的 shared memory 模板会在 attention 里反复出现。
// ============================================================================

#include <musa_runtime.h>
#include "musa_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void row_softmax_kernel(const float* x, float* y, int rows, int cols) {
    // dynamic shared memory:
    //   smem[tid] 先存每个线程扫描到的局部 max,
    //             后面复用为每个线程扫描到的局部 exp sum。
    extern __shared__ float smem[];

    // blockIdx.x 映射到矩阵的一行。这个 case 的并行粒度是:
    //   grid 维度并行不同 row, block 内线程协作处理同一个 row。
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    // 第一遍: 每个线程按 stride=blockDim.x 扫描多个列。
    // 当 cols > blockDim.x 时, 一个线程会负责 col=tid, tid+B, tid+2B...
    // 当 cols < blockDim.x 时, 多余线程得到 -INF, 不影响 max reduction。
    float local_max = -INFINITY;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_max = fmaxf(local_max, row_x[col]);
    }

    // block reduction 求整行最大值。
    // 注意每一轮都要 __syncthreads(), 因为下一轮会读取别的线程刚写的值。
    smem[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float row_max = smem[0];

    // 第二遍: 用 row_max 做数值稳定的 exp 累加。
    // 直接 exp(x) 在 x 很大时可能 overflow; exp(x - max) 最大只会是 1。
    float local_sum = 0.0f;
    for (int col = tid; col < cols; col += blockDim.x) {
        local_sum += expf(row_x[col] - row_max);
    }

    // block reduction 求整行 exp sum, 也就是 softmax 分母。
    smem[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float row_sum = smem[0];

    // 第三遍: 写回归一化结果。
    // 这里为了教学清晰重新计算 exp; 性能版可以缓存或融合后续计算。
    for (int col = tid; col < cols; col += blockDim.x) {
        row_y[col] = expf(row_x[col] - row_max) / row_sum;
    }
}

static void cpu_row_softmax(const std::vector<float>& x, std::vector<float>& y,
                            int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c) m = std::max(m, x[r * cols + c]);

        float s = 0.0f;
        for (int c = 0; c < cols; ++c) s += std::exp(x[r * cols + c] - m);

        for (int c = 0; c < cols; ++c) {
            y[r * cols + c] = std::exp(x[r * cols + c] - m) / s;
        }
    }
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, std::fabs(a[i] - b[i]));
    return err;
}

int main() {
    // 小尺寸 smoke test:
    //   rows=3, cols=8 便于肉眼看第一行输出。
    //   真正跑性能时可把 cols 改到 1024/2048 观察 reduction 成本。
    const int rows = 3;
    const int cols = 8;
    const int n = rows * cols;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n), h_y(n), h_ref(n);
    for (int i = 0; i < n; ++i) h_x[i] = (float)((i % 9) - 4) * 0.75f;
    h_x[5] = 20.0f;  // 故意放一个大值, 验证减 max 的稳定性。

    float *d_x = nullptr, *d_y = nullptr;
    MUSA_CHECK(musaMalloc(&d_x, bytes));
    MUSA_CHECK(musaMalloc(&d_y, bytes));
    MUSA_CHECK(musaMemcpy(d_x, h_x.data(), bytes, musaMemcpyHostToDevice));

    const int threads = 128;
    row_softmax_kernel<<<rows, threads, threads * sizeof(float)>>>(d_x, d_y, rows, cols);
    MUSA_CHECK_KERNEL();

    MUSA_CHECK(musaMemcpy(h_y.data(), d_y, bytes, musaMemcpyDeviceToHost));
    cpu_row_softmax(h_x, h_ref, rows, cols);

    printf("row_softmax max_abs_error=%.8f\n", max_abs_error(h_y, h_ref));
    printf("row0:");
    for (int c = 0; c < cols; ++c) printf(" %.5f", h_y[c]);
    printf("\n");

    musaFree(d_x);
    musaFree(d_y);
    return 0;
}
