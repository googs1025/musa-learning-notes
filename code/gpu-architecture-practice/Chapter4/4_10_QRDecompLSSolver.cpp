// ============================================================================
//  示例: QR 分解最小二乘求解
//
//  学习目标:
//    1. 求解过定方程 Ax ~= b, 其中 m >= n。
//    2. 使用 muSOLVER 的 QR 分解相关接口。
//    3. 关注列主序存储、leading dimension 和 workspace 查询。
//
//  注意:
//    最小二乘不是简单矩阵求逆; QR 分解通常比正规方程更稳定。
// ============================================================================
#include <iostream>
#include <musa_runtime.h>
#include <musolver.h>
int main() {
    // 问题参数：求解 Ax ≈ b 的最小二乘解
    int m = 5;   // 行数（观测数）
    int n = 3;   // 列数（变量数），要求 m >= n
    int lda = m; // 矩阵A的leading dimension

    // 初始化主机数据（列主序存储）
    float h_A[] = {1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f, 8.0f,
                   10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};

    float h_b[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

    float *h_x = new float[n]; // 解向量

    // 分配GPU内存
    float *d_A, *d_b, *d_tau;
    musaMalloc(&d_A, m * n * sizeof(float));
    musaMalloc(&d_b, m * sizeof(float));
    musaMalloc(&d_tau, n * sizeof(float));

    // 数据传输到GPU
    musaMemcpy(d_A, h_A, m * n * sizeof(float), musaMemcpyHostToDevice);
    musaMemcpy(d_b, h_b, m * sizeof(float), musaMemcpyHostToDevice);

    // 创建cuSOLVER句柄
    musolverHandle_t handle;
    musolverCreate(&handle);

    // 1. 查询工作空间大小
    int lwork = 0;
    float *d_work = nullptr;
    musolverSgeqrf_bufferSize(handle, m, n, d_A, lda, &lwork);

    // 分配工作空间
    musaMalloc(&d_work, lwork * sizeof(float));

    // 2. 执行QR分解：A = Q*R
    int *devInfo = nullptr;
    musaMalloc(&devInfo, sizeof(int));
    musolverSgeqrf(handle, m, n, d_A, lda, d_tau, d_work, lwork, devInfo);

    // 检查分解是否成功
    int h_info = 0;
    musaMemcpy(&h_info, devInfo, sizeof(int), musaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "QR分解失败，错误码: " << h_info << std::endl;
        return EXIT_FAILURE;
    }

    // 3. 应用Q^T到b：d_b <- Q^T * d_b
    musolverSormqr(handle, MUBLAS_SIDE_LEFT, MUBLAS_OP_T, m, 1, n, d_A, lda, d_tau, d_b, m, d_work,
                   lwork, devInfo);

    // 检查操作是否成功
    musaMemcpy(&h_info, devInfo, sizeof(int), musaMemcpyDeviceToHost);
    if (h_info != 0) {
        std::cerr << "应用Q^T失败，错误码: " << h_info << std::endl;
        return EXIT_FAILURE;
    }

    // 4. 从R*x = Q^T*b求解x（R是上三角矩阵）
    // 提取R矩阵的前n行（GPU内存中R存储在A的上三角部分）
    // 结果存储在d_b的前n个元素中

    // 将结果传回主机
musaMemcpy(h_x, d_b, n * sizeof(float), musaMemcpyDeviceToHost));

// 打印解向量
std::cout << "最小二乘解 x = [";
for (int i = 0; i < n; i++) {
    std::cout << h_x[i] << " ";
}
std::cout << "]" << std::endl;

// 释放资源
musaFree(d_A);
musaFree(d_b);
musaFree(d_tau);
musaFree(d_work);
musaFree(devInfo);
musolverDestroy(handle);
delete[] h_x;

return 0;
}
