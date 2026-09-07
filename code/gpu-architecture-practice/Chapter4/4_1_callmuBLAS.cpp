// ============================================================================
//  示例: 调用 muBLAS DGEMM
//
//  学习目标:
//    1. 理解 BLAS GEMM 接口: C = alpha * op(A) * op(B) + beta * C。
//    2. 区分 host vector 和 device pointer 的生命周期。
//    3. 关注 lda/ldb/ldc: BLAS 库通常按列主序语义解释矩阵。
//
//  数据流:
//    ha/hb/hc(host) -> da/db/dc(device) -> mublasDgemm -> hc(host)
//
//  阅读顺序:
//    先确认 m/n/k 与 lda/ldb/ldc, 再看 device 拷贝、GEMM 调用和结果拷回。
//
//  注意:
//    这个文件只展示库调用骨架, 没有做 hc_gold 正确性对比。用于学习时建议补
//    CPU reference 或打印小矩阵结果。
// ============================================================================
#include "mublas.h"
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <musa.h>
#include <vector>
#define DIM1 1024
#define DIM2 1024
#define DIM3 1024
int main() {
    mublasOperation_t transa = MUBLAS_OP_N, transb = MUBLAS_OP_N;
    double alpha = 1.1, beta = 0.9;
    mublas_int m = DIM1, n = DIM2, k = DIM3;
    mublas_int lda, ldb, ldc, size_a, size_b, size_c;
    int a_stride_1, a_stride_2, b_stride_1, b_stride_2;
    // lda/ldb 不是矩阵逻辑宽度, 而是相邻列在内存中的步长。
    // 当 op(A) 或 op(B) 发生转置时, leading dimension 和访问 stride 都会变化。
    if (transa == MUBLAS_OP_N) {
        lda = m;
        size_a = k * lda;
        a_stride_1 = 1;
        a_stride_2 = lda;
        mublas_cout << "N";
    } else {
        lda = k;
        size_a = m * lda;
        a_stride_1 = lda;
        a_stride_2 = 1;
        mublas_cout << "T";
    }
    if (transb == MUBLAS_OP_N) {
        ldb = k;
        size_b = n * ldb;
        b_stride_1 = 1;
        b_stride_2 = ldb;
        mublas_cout << "N: ";
    } else {
        ldb = n;
        size_b = k * ldb;
        b_stride_1 = ldb;
        b_stride_2 = 1;
        mublas_cout << "T: ";
    }
    ldc = m;
    size_c = n * ldc;
    // Naming: da is in GPU (device) memory. ha is in CPU (host) memory
    std::vector<double> ha(size_a);
    std::vector<double> hb(size_b);
    std::vector<double> hc(size_c);
    std::vector<double> hc_gold(size_c);
    // initial data on host
    srand(1);
    for (int i = 0; i < size_a; ++i) {
        ha[i] = rand() % 17;
    }
    for (int i = 0; i < size_b; ++i) {
        hb[i] = rand() % 17;
    }
    for (int i = 0; i < size_c; ++i) {
        hc[i] = rand() % 17;
    }
    hc_gold = hc;
    // allocate memory on device
    double *da, *db, *dc;
    musaMalloc(&da, size_a * sizeof(double));
    musaMalloc(&db, size_b * sizeof(double));
    musaMalloc(&dc, size_c * sizeof(double));
    // copy matrices from host to device
    musaMemcpy(da, ha.data(), sizeof(double) * size_a, musaMemcpyHostToDevice);
    musaMemcpy(db, hb.data(), sizeof(double) * size_b, musaMemcpyHostToDevice);
    musaMemcpy(dc, hc.data(), sizeof(double) * size_c, musaMemcpyHostToDevice);
    mublasHandle_t handle;
    mublasCreate(&handle);
    // BLAS GEMM 语义:
    //   dc = alpha * op(da) * op(db) + beta * dc
    // alpha/beta 是 host 标量指针; da/db/dc 是 device 矩阵指针。
    mublasDgemm(handle, transa, transb, m, n, k, &alpha, da, lda, db, ldb, &beta, dc, ldc);
    // copy output from device to CPU
    musaMemcpy(hc.data(), dc, sizeof(double) * size_c, musaMemcpyDeviceToHost);
    musaFree(da);
    musaFree(db);
    musaFree(dc);
    mublasDestroy(handle);
    return 0;
}
