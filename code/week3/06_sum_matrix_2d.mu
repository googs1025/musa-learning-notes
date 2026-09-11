// 预计输出：
//   row0=1024 expected=1024 row_last=1024
// 注意：这里每行全是 1.0，按行 atomicAdd 后每行和都应等于 W。

#include "musa_common.h"
#include <cstdio>
#include <cstdlib>

// 2D grid 示例：每个线程负责矩阵中的一个元素。
// 为了让示例短小，按行 atomicAdd 到 rows[y]；高性能版本应先 block 内归约再写出。
//
// 输入矩阵（row-major）：
//
//       column x →  0      1      2          W-1
//   row y=0      [m[0],  m[1],  m[2],  ... , m[W-1]]
//       y=1      [m[W],  m[W+1],m[W+2],... , m[2W-1]]
//        ...
//   每个线程处理 m[y * width + x]，并把它加到 rows[y]。
//
// 当前 W=1024、H=1024：
//   block = (16,16)  → 一个 block 有 16*16=256 个线程，覆盖 16 列×16 行
//   grid  = (64,64)  → 一共有 64*64=4096 个 block
//   总线程数 = 4096*256 = 1,048,576 = 1024*1024 个矩阵元素
//
// 一个 block 的覆盖关系（bx=1, by=2）：
//   x 范围 = 1*16 ... 1*16+15 = 16 ... 31（16 个连续列）
//   y 范围 = 2*16 ... 2*16+15 = 32 ... 47（16 个连续行）
//   threadIdx=(5,7) → x=1*16+5=21, y=2*16+7=39
//   该线程读取 m[39*width+21]，并累加到 rows[39]
__global__ void matrix_to_row_sums(const float* m, float* rows, int width, int height) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    // x 是列坐标，y 是行坐标；row-major 矩阵 m[y * width + x]。
    // 一个 16x16 block 覆盖 16 列 × 16 行；整个 grid 拼成 W×H 矩阵。
    // 对固定 y 的一整行，来自不同 x/block 的线程都会更新 rows[y]：
    //
    //   rows[y] = m[y*width+0] + m[y*width+1] + ...
    //             + m[y*width+width-1]
    //
    // 因此 rows[y] 是“第 y 行的和”，不是每个线程各自拥有的结果。
    // atomicAdd 保证并发更新不会互相覆盖，但也会引入原子操作竞争。
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
    // h 是 host 输入，d 是 device 输入；kernel 只能读取已拷贝到 d 的数据。
    MUSA_CHECK(musaMemcpy(d, h, W * H * sizeof(float), musaMemcpyHostToDevice));
    MUSA_CHECK(musaMemset(r, 0, H * sizeof(float)));
    // 16x16 是 2D kernel 的常见起点：索引直观，线程数 256 也比较稳。
    dim3 block(16, 16), grid((W + 15) / 16, (H + 15) / 16);
    matrix_to_row_sums<<<grid, block>>>(d, r, W, H);
    MUSA_CHECK_KERNEL();
    MUSA_CHECK(musaMemcpy(rows, r, H * sizeof(float), musaMemcpyDeviceToHost));
    std::printf("row0=%.0f expected=%d row_last=%.0f\n", rows[0], W, rows[H-1]);
    MUSA_CHECK(musaFree(d)); MUSA_CHECK(musaFree(r)); std::free(h); std::free(rows);
}
