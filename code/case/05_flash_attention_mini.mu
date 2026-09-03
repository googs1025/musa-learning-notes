// ============================================================================
//  Case 05: Flash Attention Mini
//
//  目标:
//    用最小代码展示 FlashAttention 的核心递推:
//      按 K/V tile 扫描, 不保存 scores/prob 矩阵, 在线更新 m/l/out。
//
//  注意:
//    这是教学版。为了让公式一眼能对上代码, 每个 query row 暂时由一个线程
//    顺序处理 tile。下一步优化才是把 tile 内 dot/acc 分给多个线程。
// ============================================================================

#include <musa_runtime.h>
#include "musa_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>

constexpr int TILE_N = 3;
constexpr int MAX_DIM = 16;

__global__ void flash_attention_mini_kernel(const float* q, const float* k,
                                            const float* v, float* out,
                                            int seq, int dim, float scale) {
    // 教学版映射:
    //   一个线程负责一个完整 query row。
    //   这样牺牲并行度, 但 online 更新公式最清楚。
    //
    // 后续性能版可以把 "一个线程" 升级成 "一个 block/warp",
    // 但 m/l/acc 的数学状态保持不变。
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= seq) return;

    // m: 到目前为止看过的所有 score 的最大值。
    // l: 在当前 m 坐标系下的 softmax 分母。
    // acc: 已归一化的输出向量, shape [dim]。
    float m = -INFINITY;
    float l = 0.0f;
    float acc[MAX_DIM];
    for (int d = 0; d < MAX_DIM; ++d) acc[d] = 0.0f;

    for (int tile = 0; tile < seq; tile += TILE_N) {
        int tile_end = (tile + TILE_N < seq) ? (tile + TILE_N) : seq;

        // 先计算当前 K tile 的 scores, 并得到 tile 内最大值 m_tile。
        // TILE_N 很小, 所以直接放在寄存器数组里。
        float scores[TILE_N];
        float m_tile = -INFINITY;
        for (int j = tile; j < tile_end; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot += q[row * dim + d] * k[j * dim + d];
            }
            float score = dot * scale;
            scores[j - tile] = score;
            m_tile = fmaxf(m_tile, score);
        }

        // 在 tile 自己的 m_tile 坐标系下:
        //   l_tile = sum_j exp(score_j - m_tile)
        //   acc_tile = sum_j exp(score_j - m_tile) * V_j
        float l_tile = 0.0f;
        float acc_tile[MAX_DIM];
        for (int d = 0; d < MAX_DIM; ++d) acc_tile[d] = 0.0f;

        for (int j = tile; j < tile_end; ++j) {
            float weight = expf(scores[j - tile] - m_tile);
            l_tile += weight;
            for (int d = 0; d < dim; ++d) {
                acc_tile[d] += weight * v[j * dim + d];
            }
        }

        // 把旧状态和 tile 状态合并到共同的新坐标系 m_new。
        // old_scale 是旧分母缩放后的贡献:
        //   l_old * exp(m_old - m_new)
        // tile_scale 是当前 tile 分母缩放后的贡献:
        //   l_tile * exp(m_tile - m_new)
        float m_new = fmaxf(m, m_tile);
        float old_scale = (l == 0.0f) ? 0.0f : l * expf(m - m_new);
        float tile_scale = l_tile * expf(m_tile - m_new);
        float l_new = old_scale + tile_scale;

        // acc 存的是已经除以旧 l 的归一化输出。
        // acc_tile 还没除以 l_tile, 所以合并时分别按 l_new 重新归一化。
        for (int d = 0; d < dim; ++d) {
            acc[d] = acc[d] * (old_scale / l_new)
                   + acc_tile[d] * (expf(m_tile - m_new) / l_new);
        }
        m = m_new;
        l = l_new;
    }

    // 全部 K/V tile 扫完后, acc 就等价于 softmax(QK^T)V 的一行。
    for (int d = 0; d < dim; ++d) out[row * dim + d] = acc[d];
}

static void cpu_attention(const std::vector<float>& q, const std::vector<float>& k,
                          const std::vector<float>& v, std::vector<float>& out,
                          int seq, int dim) {
    float scale = 1.0f / std::sqrt((float)dim);
    std::vector<float> scores(seq);

    for (int i = 0; i < seq; ++i) {
        for (int j = 0; j < seq; ++j) {
            float dot = 0.0f;
            for (int d = 0; d < dim; ++d) dot += q[i * dim + d] * k[j * dim + d];
            scores[j] = dot * scale;
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
    // 这个 smoke test 选择 seq=8, dim=4, TILE_N=3:
    //   seq 不能被 TILE_N 整除, 可以顺便验证最后一个不满 tile 的边界。
    const int seq = 8;
    const int dim = 4;
    const size_t bytes = seq * dim * sizeof(float);

    std::vector<float> h_q(seq * dim), h_k(seq * dim), h_v(seq * dim), h_out(seq * dim), h_ref(seq * dim);
    for (int i = 0; i < seq * dim; ++i) {
        h_q[i] = 0.11f * (float)((i % 7) - 3);
        h_k[i] = 0.13f * (float)((i % 5) - 2);
        h_v[i] = 0.17f * (float)((i % 11) - 5);
    }

    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    MUSA_CHECK(musaMalloc(&d_q, bytes));
    MUSA_CHECK(musaMalloc(&d_k, bytes));
    MUSA_CHECK(musaMalloc(&d_v, bytes));
    MUSA_CHECK(musaMalloc(&d_out, bytes));
    MUSA_CHECK(musaMemcpy(d_q, h_q.data(), bytes, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_k, h_k.data(), bytes, musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemcpy(d_v, h_v.data(), bytes, musaMemcpyHostToDevice));

    flash_attention_mini_kernel<<<1, seq>>>(d_q, d_k, d_v, d_out, seq, dim,
                                            1.0f / std::sqrt((float)dim));
    MUSA_CHECK_KERNEL();

    MUSA_CHECK(musaMemcpy(h_out.data(), d_out, bytes, musaMemcpyDeviceToHost));
    cpu_attention(h_q, h_k, h_v, h_ref, seq, dim);

    printf("flash_attention_mini max_abs_error=%.8f\n", max_abs_error(h_out, h_ref));
    printf("out row0:");
    for (int d = 0; d < dim; ++d) printf(" %.5f", h_out[d]);
    printf("\n");

    musaFree(d_q);
    musaFree(d_k);
    musaFree(d_v);
    musaFree(d_out);
    return 0;
}
