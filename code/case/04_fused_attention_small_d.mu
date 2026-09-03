// ============================================================================
//  Case 04: Fused Attention Small-D
//
//  目标:
//    一个 kernel 完成单个 query row 的 score -> softmax -> weighted V。
//
//  关键变化:
//    和 03 不同, 这里不再写出 scores[S, S] / prob[S, S]。
//    这就是 fusion 的第一步: 少写 global memory, 但 debug 难度会上升。
//
//  限制:
//    教学版固定 seq <= 64, dim <= 16。真实实现会用更细的 tiling。
// ============================================================================

#include <musa_runtime.h>
#include "musa_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int MAX_SEQ = 64;

__global__ void fused_attention_small_d_kernel(const float* q, const float* k,
                                               const float* v, float* out,
                                               int seq, int dim, float scale) {
    // shared memory 被切成两段:
    //   scores[0:MAX_SEQ]  保存当前 query row 对所有 key 的 score
    //   reduce[0:threads]  做 max/sum reduction 的临时空间
    //
    // 这已经比 naive 省内存: scores 只存在于一个 block 的 shared memory,
    // 不再把 [seq, seq] 写到 global memory。
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce = smem + MAX_SEQ;

    // 一个 block 负责一个 query row。
    // block 内前 seq 个线程分别计算一个 key 的 score。
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= seq) return;

    if (tid < seq) {
        // score_j = dot(Q[row], K[j]) / sqrt(dim)
        float acc = 0.0f;
        for (int d = 0; d < dim; ++d) {
            acc += q[row * dim + d] * k[tid * dim + d];
        }
        scores[tid] = acc * scale;
    }
    __syncthreads();

    // 对当前 query row 的所有 score 求 max。
    // tid >= seq 的线程填 -INF, 保证 reduction 形状固定为 blockDim.x。
    reduce[tid] = (tid < seq) ? scores[tid] : -INFINITY;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] = fmaxf(reduce[tid], reduce[tid + stride]);
        __syncthreads();
    }
    float m = reduce[0];

    // 求 softmax 分母 l = sum_j exp(score_j - m)。
    reduce[tid] = (tid < seq) ? expf(scores[tid] - m) : 0.0f;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) reduce[tid] += reduce[tid + stride];
        __syncthreads();
    }
    float l = reduce[0];

    // 输出维度 dim 很小, 这里让前 dim 个线程各负责一个 out[row, d]。
    // 这一步直接消费 scores, 不把 probability 写出去。
    if (tid < dim) {
        float acc = 0.0f;
        for (int j = 0; j < seq; ++j) {
            float p = expf(scores[j] - m) / l;
            acc += p * v[j * dim + tid];
        }
        out[row * dim + tid] = acc;
    }
}

static void cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, std::vector<float>& out,
                          int seq, int dim) {
    float scale = 1.0f / std::sqrt((float)dim);
    std::vector<float> scores(seq);

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) acc += q[i * dim + d] * k[j * dim + d];
            scores[j] = acc * scale;
        }
        float m = *std::max_element(scores.begin(), scores.end());
        float l = 0.0f;
        for (float s : scores) l += std::exp(s - m);
        for (int d = 0; d < dim; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < seq; ++j) acc += std::exp(scores[j] - m) / l * v[j * dim + d];
            out[i * dim + d] = acc;
        }
    }
}

static float max_abs_error(const std::vector<float>& a, const std::vector<float>& b) {
    float err = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) err = std::max(err, std::fabs(a[i] - b[i]));
    return err;
}

int main() {
    // fused 教学版的限制:
    //   seq 必须 <= MAX_SEQ, dim 需要小于 blockDim.x 且这里用 <= 16。
    // 真实 attention 会把 seq 和 dim 都继续 tile 化。
    const int seq = 8;
    const int dim = 4;
    const size_t bytes = seq * dim * sizeof(float);

    std::vector<float> h_q(seq * dim), h_k(seq * dim), h_v(seq * dim), h_out(seq * dim), h_ref(seq * dim);
    for (int i = 0; i < seq * dim; ++i) {
        h_q[i] = std::sin((float)i) * 0.5f;
        h_k[i] = std::cos((float)i) * 0.5f;
        h_v[i] = 0.25f * (float)((i % 9) - 4);
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    MUSA_CHECK(musaMalloc(&d_q, bytes));
    MUSA_CHECK(musaMalloc(&d_k, bytes));
    MUSA_CHECK(musaMalloc(&d_v, bytes));
    MUSA_CHECK(musaMalloc(&d_out, bytes));
    MUSA_CHECK(musaMemcpy(d_q, h_q.data(), bytes, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_k, h_k.data(), bytes, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_v, h_v.data(), bytes, musaMemcpyHostToDevice));

    const int threads = 128;
    size_t smem_bytes = (MAX_SEQ + threads) * sizeof(float);
    fused_attention_small_d_kernel<<<seq, threads, smem_bytes>>>(
        d_q, d_k, d_v, d_out, seq, dim, 1.0f / std::sqrt((float)dim));
    MUSA_CHECK_KERNEL();

    MUSA_CHECK(musaMemcpy(h_out.data(), d_out, bytes, musaMemcpyDeviceToHost));
    cpu_attention(h_q, h_k, h_v, h_ref, seq, dim);

    printf("fused_attention_small_d max_abs_error=%.8f\n", max_abs_error(h_out, h_ref));

    musaFree(d_q);
    musaFree(d_k);
    musaFree(d_v);
    musaFree(d_out);
    return 0;
}
