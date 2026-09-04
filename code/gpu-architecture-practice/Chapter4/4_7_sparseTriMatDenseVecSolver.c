// ============================================================================
//  示例片段: muSPARSE 稀疏三角矩阵求解
//
//  学习目标:
//    1. 用 CSR 描述稀疏三角矩阵 A。
//    2. 使用稠密向量 descriptor 表示 x/y。
//    3. 理解三角求解通常出现在 LU/ILU 分解后的前代或回代。
//
//  注意:
//    这个文件是 API 片段, 重点看 descriptor 与求解调用顺序。
// ============================================================================
#include <musparse.h>
#include <musa_runtime.h>

// 假设 A 为 n×n 稀疏三角矩阵 (CSR), x, y 为 n×1 稠密向量

// 1. 创建 musparse 句柄
musparseHandle_t handle;
musparseCreate(&handle);

// 2. 创建稀疏矩阵和向量描述符
musparseSpMatDescr_t matA;
musparseDnVecDescr_t vecX, vecY;
musparseCreateCsr(&matA, n, n, nnzA, d_A_csrRowPtr, d_A_csrColInd, d_A_csrVal, MUSA_INDEX_32I,
                  MUSA_INDEX_32I, MUSA_R_32F, MUSA_SPARSITY_GENERAL);
musparseCreateDnVec(&vecX, n, d_x, MUSA_R_32F);
musparseCreateDnVec(&vecY, n, d_y, MUSA_R_32F);

// 3. 查询缓冲区大小并分配, 进行预处理
size_t bufferSize = 0;
void *dBuffer = nullptr;
float alpha = 1.0f;

musparseSpSV(handle, MUSA_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY, MUSA_R_32F,
             MUSA_SPSV_ALG_DEFAULT, MUSPARSE_SPSV_STAGE_BUFFER_SIZE, &bufferSize, dBuffer);

musaMalloc(&dBuffer, bufferSize);

musparseSpSV(handle, MUSA_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY, MUSA_R_32F,
             MUSA_SPSV_ALG_DEFAULT, MUSPARSE_SPSV_STAGE_PREPROCESS, &bufferSize, dBuffer);

// 4. 执行SpSV计算
musparseSpSV(handle, MUSA_OPERATION_NON_TRANSPOSE, &alpha, matA, vecX, vecY, MUSA_R_32F,
             MUSA_SPSV_ALG_DEFAULT, MUSPARSE_SPSV_STAGE_COMPUTE, &bufferSize, dBuffer);

// 5. 释放资源
musparseDestroySpMat(matA);
musparseDestroyDnVec(vecX);
musparseDestroyDnVec(vecY);
musparseDestroy(handle);
musaFree(dBuffer);
