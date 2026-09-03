// ============================================================================
//  Case 02: Online Softmax
//
//  目标:
//    仍然计算 row-wise softmax, 但用 online recurrence 合并分段结果。
//
//  为什么重要:
//    FlashAttention 不会一次性保存完整 score row。它一块一块扫描 K/V,
//    每块都产生局部 max/sum, 然后用 online 公式合并到全局状态。
// ============================================================================

#include <musa_runtime.h>
#include "musa_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

struct SoftmaxPair {
    float m;  // 当前片段最大值
    float l;  // sum(exp(x_i - m)), 注意它依赖当前 m
};

__device__ SoftmaxPair combine_pair(SoftmaxPair a, SoftmaxPair b) {
    // 两段 softmax 状态不能直接 l 相加, 因为它们各自减掉的 max 不同。
    // 先切到共同的新 max, 再把两个 sum 缩放到同一个坐标系。
    float m = fmaxf(a.m, b.m);
    float l = a.l * expf(a.m - m) + b.l * expf(b.m - m);
    return {m, l};
}

__global__ void online_softmax_kernel(const float* x, float* y, int rows, int cols) {
    // partial[tid] 保存每个线程扫描自己列片段得到的 (m, l)。
    // 后续 reduction 合并的是 SoftmaxPair, 不是单个 float。
    extern __shared__ SoftmaxPair partial[];

    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= rows) return;

    const float* row_x = x + row * cols;
    float* row_y = y + row * cols;

    // 每个线程顺序读若干元素, 用 combine_pair 把单个元素并入本线程状态。
    // 单个元素 x 的状态是 (m=x, l=1), 因为 exp(x-x)=1。
    SoftmaxPair local{-INFINITY, 0.0f};
    for (int col = tid; col < cols; col += blockDim.x) {
        SoftmaxPair one{row_x[col], 1.0f};
        local = combine_pair(local, one);
    }

    // block 内 pair reduction。合并完 partial[0] 就是整行的最终 (m, l)。
    partial[tid] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) partial[tid] = combine_pair(partial[tid], partial[tid + stride]);
        __syncthreads();
    }

    float m = partial[0].m;
    float l = partial[0].l;
    // 输出阶段仍然要再读一遍 x, 因为 online softmax 只保存了全局 m/l。
    // FlashAttention 会把这一步替换成对 V 的加权累加。
    for (int col = tid; col < cols; col += blockDim.x) {
        row_y[col] = expf(row_x[col] - m) / l;
    }
}

static void cpu_row_softmax(const std::vector<float>& x, std::vector<float>& y,
                            int rows, int cols) {
    for (int r = 0; r < rows; ++r) {
        float m = -INFINITY;
        for (int c = 0; c < cols; ++c) m = std::max(m, x[r * cols + c]);

        float l = 0.0f;
        for (int c = 0; c < cols; ++c) l += std::exp(x[r * cols + c] - m);

        for (int c = 0; c < cols; ++c) y[r * cols + c] = std::exp(x[r * cols + c] - m) / l;
    }
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, std::fabs(a[i] - b[i]));
    return err;
}

int main() {
    const int rows = 2;
    const int cols = 17;
    const int n = rows * cols;
    const size_t bytes = n * sizeof(float);

    std::vector<float> h_x(n), h_y(n), h_ref(n);
    for (int i = 0; i < n; ++i) h_x[i] = std::sin((float)i) * 3.0f;

    float *d_x = nullptr, *d_y = nullptr;
    MUSA_CHECK(musaMalloc(&d_x, bytes));
    MUSA_CHECK(musaMalloc(&d_y, bytes));
    MUSA_CHECK(musaMemcpy(d_x, h_x.data(), bytes, musaMemcpyHostToDevice));

    const int threads = 128;
    online_softmax_kernel<<<rows, threads, threads * sizeof(SoftmaxPair)>>>(d_x, d_y, rows, cols);
    MUSA_CHECK_KERNEL();

    MUSA_CHECK(musaMemcpy(h_y.data(), d_y, bytes, musaMemcpyDeviceToHost));
    cpu_row_softmax(h_x, h_ref, rows, cols);

    printf("online_softmax max_abs_error=%.8f\n", max_abs_error(h_y, h_ref));

    musaFree(d_x);
    musaFree(d_y);
    return 0;
}
