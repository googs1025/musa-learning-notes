// ============================================================================
//  示例: GEMV 优化片段
//
//  学习目标:
//    1. 用 warp/group 内 shuffle 完成局部规约。
//    2. 理解 GEMV 的算术强度低, 性能通常更受内存带宽限制。
//    3. 观察向量化加载、线程协作和规约 mask 如何影响吞吐。
//
//  注意:
//    这个文件偏 kernel 优化片段, 不是最小入门示例。建议先读 week5 的 naive GEMM
//    和 reduction 示例, 再回来看这里的 shuffle 规约。
//
//  阅读顺序:
//    先看 shuffle 的 width/mask, 再看 warp reduction, 最后看 GEMV 的 tile 和向量化加载。
// ============================================================================
#define WARP_THREADS 32
// SHFL_MASK
#define MASK_SHFL_32 ((~(32 - 1)) & 0x7f) << 7 | (32 - 1)
#define MASK_SHFL_16 ((~(16 - 1)) & 0x7f) << 7 | (16 - 1)
#define MASK_SHFL_8 ((~(8 - 1)) & 0x7f) << 7 | (8 - 1)
#define MASK_SHFL_4 ((~(4 - 1)) & 0x7f) << 7 | (4 - 1)
#define MASK_SHFL_2 ((~(2 - 1)) & 0x7f) << 7 | (2 - 1)

template <typename T, int width>
__device__ __forceinline__ T shfl_down_sync(T val, unsigned int delta) {
    // shuffle 直接在线程之间交换寄存器值, 不需要经过 shared memory。
    int ret = 0;
    int tmp = *(reinterpret_cast<int32_t *>(&val));
    if constexpr (width == 32) {
        ret = __musa_shfl_down_sync_i32(tmp, delta & 0x1f, MASK_SHFL_32);
    } else if constexpr (width == 16) {
        ret = __musa_shfl_down_sync_i32(tmp, delta & 0xf, MASK_SHFL_16);
    } else if constexpr (width == 8) {
        ret = __musa_shfl_down_sync_i32(tmp, delta & 0x7, MASK_SHFL_8);
    } else if constexpr (width == 4) {
        ret = __musa_shfl_down_sync_i32(tmp, delta & 0x3, MASK_SHFL_4);
    } else if constexpr (width == 2) {
        ret = __musa_shfl_down_sync_i32(tmp, delta & 0x1, MASK_SHFL_2);
    }
    return *(reinterpret_cast<T *>(&ret));
}

template <typename T, int width> __device__ __forceinline__ T shfl_idx_sync(T val, int src_lane) {
    int ret = 0;
    int tmp = *(reinterpret_cast<int32_t *>(&val));
    if constexpr (width == 32) {
        ret = __musa_shfl_idx_sync_i32(tmp, src_lane & 0x1f, MASK_SHFL_32);
    } else if constexpr (width == 16) {
        ret = __musa_shfl_idx_sync_i32(tmp, src_lane & 0xf, MASK_SHFL_16);
    } else if constexpr (width == 8) {
        ret = __musa_shfl_idx_sync_i32(tmp, src_lane & 0x7, MASK_SHFL_8);
    } else if constexpr (width == 4) {
        ret = __musa_shfl_idx_sync_i32(tmp, src_lane & 0x3, MASK_SHFL_4);
    } else if constexpr (width == 2) {
        ret = __musa_shfl_idx_sync_i32(tmp, src_lane & 0x1, MASK_SHFL_2);
    }
    return *(reinterpret_cast<T *>(&ret));
}

template <typename T, int blockSize> __device__ __forceinline__ void WarpReduce(T &rv1) {
    // 每轮把更高 lane 的部分和搬到当前线程, 最终 lane 0 持有小组总和。
    T rv2;
    if constexpr (blockSize >= 32) {
        rv2 = shfl_down_sync<float, 32>(rv1, 16);
        rv1 += rv2;
    }
    if constexpr (blockSize >= 16) {
        rv2 = shfl_down_sync<float, 16>(rv1, 8);
        rv1 += rv2;
    }
    if constexpr (blockSize >= 8) {
        rv2 = shfl_down_sync<float, 8>(rv1, 4);
        rv1 += rv2;
    }
    if constexpr (blockSize >= 4) {
        rv2 = shfl_down_sync<float, 4>(rv1, 2);
        rv1 += rv2;
    }
    if constexpr (blockSize >= 2) {
        rv2 = shfl_down_sync<float, 2>(rv1, 1);
        rv1 += rv2;
    }
}

template <int BLOCK_M, int BLOCK_K, int TILE_V>
__global__ void sgemv_kernel(float *o, const float *mat, const float *v, const int m, const int k) {
    using vec = float4;
    constexpr int Vlen = 4;

    int bx = blockIdx.x;
    int tid = threadIdx.x;
    if (bx * BLOCK_M >= m)
        return;

    constexpr int BLOCK_K_DIV_VLEN = BLOCK_K / Vlen;

    // 每个线程处理一个向量化加载位置; slice_id 标识输出行, lane_id 标识行内片段。
    int slice_id = tid / (BLOCK_K_DIV_VLEN);
    int lane_id = tid % (BLOCK_K_DIV_VLEN);

    int m_id = bx * BLOCK_M + slice_id;
    float m_reg[Vlen];
    float res = float(0);

    // shared memory for staging vector
    __shared__ float __attribute__((aligned(16))) smem_v[TILE_V];

    int outer_kiters = (k + TILE_V - 1) / (TILE_V);
    constexpr int inner_kiters = TILE_V / (BLOCK_K);
    float cur_thread_sum = 0;

    for (int i = 0; i < outer_kiters; i++) {
        for (int j = tid; j < TILE_V; j += blockDim.x) {
            int v_offset = i * TILE_V + j;
            smem_v[j] = v_offset < k ? v[v_offset] : float(0);
        }
        // 确保整个 block 都完成 v tile 加载后, 才允许线程读取 shared memory。
        __syncthreads();

        int s_v_id = lane_id * Vlen;
        int k_idx = i * TILE_V + s_v_id;

        if (m_id < m) {
            for (int j = 0; j < inner_kiters && k_idx < k;
                 j++, s_v_id += BLOCK_K, k_idx += BLOCK_K) {
                int64_t base_mat_offset = (int64_t)m_id * k + (int64_t)k_idx;
                if (k_idx + Vlen <= k) {
                    *((vec *)m_reg) = *(vec *)(mat + base_mat_offset);
                } else {
                    m_reg[0] = k_idx < k ? *(mat + base_mat_offset) : (float)0;
#pragma unroll
                    for (int p = 1; p < Vlen; p++) {
                        m_reg[p] = k_idx + p < k ? *(mat + base_mat_offset + p) : (float)0;
                    }
                }

                float v_reg[Vlen];
                *(vec *)(v_reg) = *(vec *)(smem_v + s_v_id);
#pragma unroll
                for (int p = 0; p < Vlen; p++) {
                    cur_thread_sum += (float)v_reg[p] * (float)m_reg[p];
                }
            }
        }
        __syncthreads();
    }

    constexpr int nr_warp_per_blockx = BLOCK_K_DIV_VLEN / WARP_THREADS;
    int warp_id = tid % BLOCK_K_DIV_VLEN / WARP_THREADS;
    constexpr int REDUCE_BLOCK = BLOCK_K_DIV_VLEN > WARP_THREADS ? WARP_THREADS : BLOCK_K_DIV_VLEN;

    // 先做 warp 内规约; 如果一个输出行跨多个 warp, 后面再用 shared memory 合并。
    WarpReduce<float, REDUCE_BLOCK>(cur_thread_sum);
    float sum = shfl_idx_sync<float, REDUCE_BLOCK>(cur_thread_sum, 0);

    // inter-warp reduction if needed
    if constexpr (nr_warp_per_blockx > 1) {
        __shared__ float smem_shfl[BLOCK_M][nr_warp_per_blockx];
        smem_shfl[slice_id][warp_id] = sum;
        __syncthreads();
        if (warp_id == 0) {
            for (int i = 1; i < nr_warp_per_blockx; i++) {
                sum += smem_shfl[slice_id][i];
            }
            smem_shfl[slice_id][0] = sum;
        }
        __syncthreads();
        sum = smem_shfl[slice_id][0];
    }

    if (lane_id == 0 && m_id < m) {
        o[m_id] = sum;
    }
}
