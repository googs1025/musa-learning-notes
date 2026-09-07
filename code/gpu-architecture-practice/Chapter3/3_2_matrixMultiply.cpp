// ============================================================================
//  示例: 3x3 整数矩阵乘法
//
//  学习目标:
//    1. 用二维 threadIdx/blockIdx 映射矩阵的 row/col。
//    2. 理解一个线程计算 C[row, col] 的完整 dot product。
//    3. 建立 naive GEMM 的最小数据流: A/B 从 host 拷到 device, C 再拷回 host。
//
//  阅读顺序:
//    先确认 row/col 映射, 再沿着 k 循环理解一个输出元素的点积。
//
//  注意:
//    这个示例为了清楚只使用一个 block 和 3x3 线程, 不是高性能 GEMM 写法。
//    后续可和 week5 的 tiled GEMM / muBLAS 示例对照。
// ============================================================================
#include <stdio.h>
#include <musa_runtime.h>

#define N 3

// MUSA核函数：矩阵乘法
__global__ void matrixMultiply(int *a, int *b, int *c, int n) {
    // 二维 block/grid 映射到矩阵坐标: y 方向负责行, x 方向负责列。
    // 对更大的矩阵, blocksPerGrid 也会扩展成二维。
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        int sum = 0;
        for (int k = 0; k < n; k++) {
            // row 固定、col 固定, k 扫过公共维度:
            // C[row, col] = sum_k A[row, k] * B[k, col]。
            sum += a[row * n + k] * b[k * n + col];
        }
        c[row * n + col] = sum;
    }
}

int main() {
    // 定义并初始化主机内存中的矩阵A和B
    int h_A[N][N] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    int h_B[N][N] = {{0, 1, 2}, {3, 4, 5}, {6, 7, 8}};
    int h_C[N][N] = {0}; // 结果矩阵

    // 定义设备内存指针
    int *d_A, *d_B, *d_C;

    // 分配设备内存
    musaMalloc((void **)&d_A, N * N * sizeof(int));
    musaMalloc((void **)&d_B, N * N * sizeof(int));
    musaMalloc((void **)&d_C, N * N * sizeof(int));

    // 将数据从主机复制到设备
    musaMemcpy(d_A, h_A, N * N * sizeof(int), musaMemcpyHostToDevice);
    musaMemcpy(d_B, h_B, N * N * sizeof(int), musaMemcpyHostToDevice);

    // 定义网格和块尺寸。这里 N 很小, 一个 block 就覆盖整个输出矩阵。
    // 大矩阵需要用 ceil 除法计算 blocksPerGrid.x/y, 并保留 kernel 内边界判断。
    dim3 threadsPerBlock(N, N); // 每个块有N×N个线程
    dim3 blocksPerGrid(1, 1);   // 网格有1×1个块

    // 启动核函数
    matrixMultiply<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 等待所有线程完成
    musaDeviceSynchronize();

    // 将结果从设备复制回主机
    musaMemcpy(h_C, d_C, N * N * sizeof(int), musaMemcpyDeviceToHost);

    // 打印结果
    printf("Result matrix C = A * B:\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_C[i][j]);
        }
        printf("\n");
    }

    // 释放设备内存
    musaFree(d_A);
    musaFree(d_B);
    musaFree(d_C);

    return 0;
}
