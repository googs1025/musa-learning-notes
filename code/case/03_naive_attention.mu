// ============================================================================
//  Case 03: Naive Attention
//
//  目标:
//    用三个 kernel 明确实现 attention:
//      scores = Q K^T / sqrt(D)
//      prob   = softmax(scores)
//      out    = prob V
//
//  这个版本会落下 scores[S, S] 和 prob[S, S], 性能不是重点。
//  它的价值是最容易检查中间结果, 适合作为 fused/flash 版本的 reference。
// ============================================================================

#include <musa_runtime.h>
#include "musa_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

__global__ void qk_scores_kernel(const float* q, const float* k, float* scores,
                                 int seq, int dim, float scale) {
    // 一个线程负责 scores[row, col]。
    // row 是 query token, col 是 key token。
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq || col >= seq) return;

    // K 在内存中按 [seq, dim] row-major 存, K^T 只是访问模式:
    //   K^T[d, col] == K[col, d]
    // 不需要真的做一次 transpose。
    float acc = 0.0f;
    for (int d = 0; d < dim; ++d) {
        acc += q[row * dim + d] * k[col * dim + d];
    }
    scores[row * seq + col] = acc * scale;
}

__global__ void softmax_rows_kernel(const float* x, float* y, int rows, int cols) {
    extern __shared__ float smem[];
    int row = blockIdx.x;
    int tid = threadIdx.x;

    // 这里复用 Case 01 的稳定 softmax 模板。
    // 在 attention 里 rows=seq, cols=seq, 每行对应一个 query 对所有 key 的概率分布。
    float local_max = -INFINITY;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_max = fmaxf(local_max, x[row * cols + c]);
    }
    smem[tid] = local_max;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] = fmaxf(smem[tid], smem[tid + stride]);
        __syncthreads();
    }
    float m = smem[0];

    float local_sum = 0.0f;
    for (int c = tid; c < cols; c += blockDim.x) {
        local_sum += expf(x[row * cols + c] - m);
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) smem[tid] += smem[tid + stride];
        __syncthreads();
    }
    float l = smem[0];

    for (int c = tid; c < cols; c += blockDim.x) {
        y[row * cols + c] = expf(x[row * cols + c] - m) / l;
    }
}

__global__ void pv_kernel(const float* prob, const float* v, float* out,
                          int seq, int dim) {
    // 一个线程负责 out[row, d]。
    // 它需要遍历所有 key/value token j, 做 sum_j prob[row,j] * V[j,d]。
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= seq || d >= dim) return;

    float acc = 0.0f;
    for (int j = 0; j < seq; ++j) {
        acc += prob[row * seq + j] * v[j * dim + d];
    }
    out[row * dim + d] = acc;
}

static void cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, std::vector<float>& out,
                          int seq, int dim) {
    std::vector<float> scores(seq * seq), prob(seq * seq);
    float scale = 1.0f / std::sqrt((float)dim);

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < dim; ++d) acc += q[i * dim + d] * k[j * dim + d];
            scores[i * seq + j] = acc * scale;
        }

        float m = -INFINITY;
        for (int j = 0; j < seq; ++j) m = std::max(m, scores[i * seq + j]);
        float l = 0.0f;
        for (int j = 0; j < seq; ++j) l += std::exp(scores[i * seq + j] - m);
        for (int j = 0; j < seq; ++j) prob[i * seq + j] = std::exp(scores[i * seq + j] - m) / l;
    }

    for (int i = 0; i < seq; ++i) {
        for (int d = 0; d < dim; ++d) {
            float acc = 0.0f;
            for (int j = 0; j < seq; ++j) acc += prob[i * seq + j] * v[j * dim + d];
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
    // 小矩阵便于和 CPU reference 对齐:
    //   Q/K/V shape 都是 [seq, dim]。
    //   scores/prob shape 是 [seq, seq], 这也是 naive attention 的内存痛点。
    const int seq = 4;
    const int dim = 4;
    const size_t bytes_qkv = seq * dim * sizeof(float);
    const size_t bytes_mat = seq * seq * sizeof(float);

    std::vector<float> h_q(seq * dim), h_k(seq * dim), h_v(seq * dim), h_out(seq * dim), h_ref(seq * dim);
    for (int i = 0; i < seq * dim; ++i) {
        h_q[i] = 0.1f * (float)((i % 7) - 3);
        h_k[i] = 0.2f * (float)((i % 5) - 2);
        h_v[i] = 0.3f * (float)((i % 6) - 1);
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr;
    float *d_scores = nullptr, *d_prob = nullptr, *d_out = nullptr;
    MUSA_CHECK(musaMalloc(&d_q, bytes_qkv));
    MUSA_CHECK(musaMalloc(&d_k, bytes_qkv));
    MUSA_CHECK(musaMalloc(&d_v, bytes_qkv));
    MUSA_CHECK(musaMalloc(&d_scores, bytes_mat));
    MUSA_CHECK(musaMalloc(&d_prob, bytes_mat));
    MUSA_CHECK(musaMalloc(&d_out, bytes_qkv));
    MUSA_CHECK(musaMemcpy(d_q, h_q.data(), bytes_qkv, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_k, h_k.data(), bytes_qkv, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_v, h_v.data(), bytes_qkv, musaMemcpyHostToDevice));

    dim3 block2d(16, 16);
    dim3 grid_scores((seq + block2d.x - 1) / block2d.x, (seq + block2d.y - 1) / block2d.y);
    qk_scores_kernel<<<grid_scores, block2d>>>(d_q, d_k, d_scores, seq, dim, 1.0f / std::sqrt((float)dim));
    MUSA_CHECK_KERNEL();

    const int threads = 128;
    softmax_rows_kernel<<<seq, threads, threads * sizeof(float)>>>(d_scores, d_prob, seq, seq);
    MUSA_CHECK_KERNEL();

    dim3 grid_out((dim + block2d.x - 1) / block2d.x, (seq + block2d.y - 1) / block2d.y);
    pv_kernel<<<grid_out, block2d>>>(d_prob, d_v, d_out, seq, dim);
    MUSA_CHECK_KERNEL();

    MUSA_CHECK(musaMemcpy(h_out.data(), d_out, bytes_qkv, musaMemcpyDeviceToHost));
    cpu_attention(h_q, h_k, h_v, h_ref, seq, dim);

    printf("naive_attention max_abs_error=%.8f\n", max_abs_error(h_out, h_ref));
    printf("out row0:");
    for (int d = 0; d < dim; ++d) printf(" %.5f", h_out[d]);
    printf("\n");

    musaFree(d_q);
    musaFree(d_k);
    musaFree(d_v);
    musaFree(d_scores);
    musaFree(d_prob);
    musaFree(d_out);
    return 0;
}
