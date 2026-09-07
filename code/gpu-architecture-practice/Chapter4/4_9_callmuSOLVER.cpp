// ============================================================================
//  示例: 调用 muSOLVER 解线性方程组
//
//  学习目标:
//    1. 创建 musolverHandle_t 并绑定 stream。
//    2. 查询分解/求解需要的 workspace。
//    3. 把矩阵和右端项拷到 device, 调用 solver, 再拷回结果。
//
//  阅读顺序:
//    先看 handle/stream 生命周期, 再看 workspace、分解调用和 devInfo 错误检查。
//
//  注意:
//    solver 类 API 一般会返回 info/devInfo, 必须检查它来判断数值分解是否成功。
// ============================================================================
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <musa_runtime.h>
#include <musolver.h>
int main(int argc, char *argv[]) {
    musolverHandle_t musolverH = NULL;
    musaStream_t stream = NULL;
    const int m = 3;
    const int lda = m;
    const int ldb = m;
    /*
     *       | 1 2 3  |
     *   A = | 4 5 6  |
     *       | 7 8 10 |
     *
     * without pivoting: A = L*U
     *       | 1 0 0 |      | 1  2  3 |
     *   L = | 4 1 0 |, U = | 0 -3 -6 |
     *       | 7 2 1 |      | 0  0  1 |
     *
     * with pivoting: P*A = L*U
     *       | 0 0 1 |
     *   P = | 1 0 0 |
     *       | 0 1 0 |
     *
     *       | 1       0     0 |      | 7  8       10     |
     *   L = | 0.1429  1     0 |, U = | 0  0.8571  1.5714 |
     *       | 0.5714  0.5   1 |      | 0  0       -0.5   |
     */
    const std::vector<float> A = {1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 10.0};
    const std::vector<float> B = {1.0, 2.0, 3.0};
    std::vector<float> X(m, 0);
    std::vector<float> LU(lda * m, 0);
    std::vector<int> Ipiv(m, 0);
    int info = 0;
    float *d_A = nullptr;    /* device copy of A */
    float *d_B = nullptr;    /* device copy of B */
    int *d_Ipiv = nullptr;   /* pivoting sequence */
    int *d_info = nullptr;   /* error info */
    int lwork = 0;           /* size of workspace */
    float *d_work = nullptr; /* device workspace for getrf */
    const int pivot_on = 0;
    if (pivot_on) {
        printf("pivot is on : compute P*A = L*U \n");
    } else {
        printf("pivot is off: compute A = L*U (not numerically stable)\n");
    }
    /* step 1: create musolver handle, bind a stream */
    musolverCreate(&musolverH);
    musaStreamCreateWithFlags(&stream, musaStreamNonBlocking);
    musolverSetStream(musolverH, stream);
    /* step 2: copy A to device */
    musaMalloc(reinterpret_cast<void **>(&d_A), sizeof(float) * A.size());
    musaMalloc(reinterpret_cast<void **>(&d_B), sizeof(float) * B.size());
    musaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int) * Ipiv.size());
    musaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int));
         musaMemcpyAsync(d_A, A.data(), sizeof(float) * A.size(), musaMemcpyHostToDevice, stream);
         musaMemcpyAsync(d_B, B.data(), sizeof(float) * B.size(), musaMemcpyHostToDevice, stream);
    /* step 3: query working space of getrf */
    musolverDgetrf_bufferSize(musolverH, m, m, d_A, lda, &lwork);
    musaMalloc(reinterpret_cast<void **>(&d_work), sizeof(float) * lwork);
    /* step 4: LU factorization */
    if (pivot_on) {
        musolverSgetrf(musolverH, m, m, d_A, lda, d_work, d_Ipiv, d_info);
    } else {
        musolverSgetrf(musolverH, m, m, d_A, lda, d_work, NULL, d_info);
    }
    if (pivot_on) {
        musaMemcpyAsync(Ipiv.data(), d_Ipiv, sizeof(int) * Ipiv.size(), musaMemcpyDeviceToHost,
                        stream);
    }
         musaMemcpyAsync(LU.data(), d_A, sizeof(float) * A.size(), musaMemcpyDeviceToHost, stream);
    musaMemcpyAsync(&info, d_info, sizeof(int), musaMemcpyDeviceToHost, stream);
    musaStreamSynchronize(stream);
    if (0 > info) {
        printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }
    if (pivot_on) {
        printf("pivoting sequence, matlab base-1\n");
        for (int j = 0; j < m; j++) {
            printf("Ipiv(%d) = %d\n", j + 1, Ipiv[j]);
        }
    }
    /*
     * step 5: solve A*X = B
     *       | 1 |       | -0.3333 |
     *   B = | 2 |,  X = |  0.6667 |
     *       | 3 |       |  0      |
     *
     */
    if (pivot_on) {
        musolverSgetrs(musolverH, MUBLAS_OP_N, m, 1, /* nrhs */
                       d_A, lda, d_Ipiv, d_B, ldb, d_info);
    } else {
        musolverSgetrs(musolverH, MUBLAS_OP_N, m, 1, /* nrhs */
                       d_A, lda, NULL, d_B, ldb, d_info);
    }
    musaMemcpyAsync(X.data(), d_B, sizeof(float) * X.size(), musaMemcpyDeviceToHost, stream);
    musaStreamSynchronize(stream);
    /* free resources */
    musaFree(d_A);
    musaFree(d_B);
    musaFree(d_Ipiv);
    musaFree(d_info);
    musaFree(d_work);
    musolverDestroy(musolverH);
    musaStreamDestroy(stream);
    musaDeviceReset();
    return 0;
}
