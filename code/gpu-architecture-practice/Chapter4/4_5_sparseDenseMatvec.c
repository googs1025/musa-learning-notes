// ============================================================================
//  示例片段: muSPARSE SpMV, 稀疏矩阵乘稠密向量
//
//  学习目标:
//    1. 用 CSR 三数组描述稀疏矩阵 A。
//    2. 用 dense vector descriptor 描述输入 x 和输出 y。
//    3. 通过 bufferSize -> malloc buffer -> compute 的顺序调用 SpMV。
//
//  注意:
//    这个文件是 API 片段, m/n/nnz/d_* 等变量需要由完整程序提供。
// ============================================================================
#include <musparse.h>
#include <musa_runtime.h>

// 假设 A 为 m × n 的稀疏矩阵 (CSR), x 为 n × 1，y 为 m × 1 向量

// 阅读顺序: 先看 CSR rowPtr/colInd/value 三数组如何描述 A, 再看 SpMV 的临时 buffer 生命周期。

// 1. 创建 musparse 句柄
musparseHandle_t handle;
musparseCreate(&handle);

// 2. 创建稀疏矩阵描述符
musparseSpMatDescr_t matA;
musparseCreateCsr(&matA, m, n, nnz, d_csrRowPtr, d_csrColInd, d_csrVal, MUSA_INDEX_32I,
                  MUSA_INDEX_32I, MUSA_R_32F, MUSA_SPARSITY_GENERAL);

// 3. 创建向量描述符
musparseDnVecDescr_t vecX, vecY;
musparseCreateDnVec(&vecX, n, d_x, MUSA_R_32F);
musparseCreateDnVec(&vecY, m, d_y, MUSA_R_32F);

// 4. 查询缓冲区大小并分配
float alpha = 1.0f, beta = 0.0f;
size_t bufferSize = 0;
void *dBuffer = nullptr;

musparseSpMV(handle, MUSA_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, MUSA_R_32F,
             MUSA_SPMV_ALG_DEFAULT, &bufferSize, dBuffer);

musaMalloc(&dBuffer, bufferSize);

// 5. 执行SpMV计算
musparseSpMV(handle, MUSA_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, &beta, vecY, MUSA_R_32F,
             MUSA_SPMV_ALG_DEFAULT, &bufferSize, dBuffer);

// 6. 释放资源
musparseDestroySpMat(matA);
musparseDestroyDnVec(vecX);
musparseDestroyDnVec(vecY);
musparseDestroy(handle);
musaFree(dBuffer);
