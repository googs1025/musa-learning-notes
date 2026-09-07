// ============================================================================
//  示例: GEMM 优化伪代码
//
//  学习目标:
//    1. 理解 tiled GEMM 的多级数据搬运: global -> register -> shared -> register。
//    2. 使用 shared memory 双缓冲隐藏部分 global memory 访问延迟。
//    3. 用寄存器 tile 累加多个 C 元素, 提高数据复用。
//
//  注意:
//    文件名是 GEMM, 但代码中包含省略号和占位函数, 属于优化结构草图, 不能直接编译。
//    读它时关注流水线顺序, 不要把它当完整 kernel。
//
//  阅读顺序:
//    先看 shared tile 的双缓冲, 再看 global/shared/register 三层数据搬运和寄存器累加。
// ============================================================================
__global__ void mat_mul_pseudo(float *output, const float *input_a, const float *input_b, int m,
                               int n, int k) {
    constexpr int tile_size = 128;
    constexpr int TILE_K = 4;

    // ping pong switch is optional
    // shared memory, 2row for double buffer
    __shared__ float __attribute__((aligned(16))) smem_a[2][TILE_K * tile_size];
    __shared__ float __attribute__((aligned(16))) smem_b[2][TILE_K * tile_size];

    // register
    float4 load_reg_a[2];
    float4 load_reg_b[2];
    float4 reg_a[2][2]; // 2row for double buffer
    float4 reg_b[2][2];
    float4 reg_c[8][2] = {{float4(0.f, 0.f, 0.f, 0.f)}};

    swizzle_blkid(blockIdx.x, blockIdx.y, blockIdx.z);
    // offset index compute
    int load_offset_a = ...;
    int load_offset_b = ...;
    int store_offset_a = ...;
    int store_offset_b = ...;

    // load first block from global memory to shared memory
    load_gmem_to_reg(input_a, load_offset_a, load_reg_a);
    load_gmem_to_reg(input_b, load_offset_b, load_reg_b);
    store_reg_to_smem_with_transpose(load_reg_a, store_offset_a, smem_a[0]);
    store_reg_to_smem(load_reg_b, store_offset_b, smem_b[0]);
    __syncthreads();

    // load first register from shared memory
    load_smem_to_reg(smem_a[0], 0, reg_a[0]);
    load_smem_to_reg(smem_b[0], 0, reg_b[0]);

    for (int i = TILE_K; i < k + TILE_K; i += TILE_K) {
        // load data from global memory
        load_gmem_to_reg(input_a, load_offset_a + i, load_reg_a);
        load_gmem_to_reg(input_b, load_offset_b + i, load_reg_b);

        for (int j = 0; j < TILE_K - 1; j++) {
            // load data from shared memory to register for next iteration
            load_smem_to_reg(smem_a[(i - TILE_K) / TILE_K], j + 1, reg_a[(j + 1) % 2]);
            load_smem_to_reg(smem_b[(i - TILE_K) / TILE_K], j + 1, reg_b[(j + 1) % 2]);
            // compute matrix multiply accumulate 8x8
            mma8x8(reg_c, reg_a[j % 2], reg_c[j % 2]);
        }

        if (i < k) {
            // make sure that the computation is done before the next iteration
            __syncthreads();
            // store data to shared memory before the next iteration
            store_reg_to_smem_with_transpose(load_reg_a, store_offset_a, smem_a[i / TILE_K]);
            store_reg_to_smem(load_reg_b, store_offset_b, smem_b[i / TILE_K]);
            __syncthreads();
        }

        // load data from shared memory to register before the next iteration
        load_smem_tile_to_reg(smem_a[i / TILE_K], 0, a_reg[0]);
        load_smem_tile_to_reg(smem_b[i / TILE_K], 0, b_reg[0]);
        // compute the last matrix multiply accumulate 8x8
        mma8x8(reg_c, a_reg[1], b_reg[1]);
    }
    // store the result register to global memory
    store_reg_to_gmem(output, reg_c);
}
