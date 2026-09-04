// ============================================================================
//  示例片段: muSPARSE 稀疏向量点积
//
//  学习目标:
//    1. 用 sparse vector descriptor 表达只存非零值的向量 x。
//    2. 用 dense vector descriptor 表达普通向量 y。
//    3. 调用 musparseSpVV 计算 x dot y。
//
//  注意:
//    这个文件是 API 片段, n/nnz/d_* 等变量需要由完整程序提供。
// ============================================================================
#include <musparse.h>
#include <musa_runtime.h>

// 假设 x 是 n 维稀疏向量（如 COO 格式），y 是 n 维稠密向量

// 1. 创建 musparse 句柄
musparseHandle_t handle;
musparseCreate(&handle);

// 2. 创建稀疏向量和稠密向量描述符
musparseSpVecDescr_t vecX;
musparseDnVecDescr_t vecY;
musparseCreateSpVec(&vecX, n, nnzX, d_x_indices, d_x_values, MUSA_INDEX_32I, MUSA_R_32F);
musparseCreateDnVec(&vecY, n, d_y, MUSA_R_32F);

// 3. 查询缓冲区大小并分配
float result = 0.0f;
size_t bufferSize = 0;
void *dBuffer = nullptr;

musparseSpVV(handle, MUSA_OPERATION_NON_TRANSPOSE, vecX, vecY, &result, MUSA_R_32F, &bufferSize,
             dBuffer);

CHECK_MUSA_ERROR(musaMalloc(&dBuffer, bufferSize));

// 4. 执行 SpVV（稀疏·稠密向量点积）
musparseSpVV(handle, MUSA_OPERATION_NON_TRANSPOSE, vecX, vecY, &result, MUSA_R_32F, &bufferSize,
             dBuffer);

// 5. 释放资源
musparseDestroySpVec(vecX);
musparseDestroyDnVec(vecY);
musparseDestroy(handle);
