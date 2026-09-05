// ============================================================================
//  示例片段: muSPARSE SpGEMM, 稀疏矩阵乘稀疏矩阵
//
//  学习目标:
//    1. 理解 SpGEMM 需要先估计/生成输出 C 的稀疏结构。
//    2. 区分 workEstimation、compute、copy 三个阶段。
//    3. 观察 CSR descriptor 如何贯穿 A/B/C/D。
//
//  注意:
//    这个文件是 API 片段, 省略了输入矩阵初始化和 descriptor 销毁。
// ============================================================================
#include <musparse.h>
#include <musa_runtime.h>

// 假设 A, B, D, C 都为 CSR 格式稀疏矩阵

// 1. 创建 musparse 句柄
musparseHandle_t handle;
musparseCreate(&handle);

musaDataType computeType = MUSA_R_32F;
musparseIndexType_t indexType = MUSPARSE_INDEX_32I;

// 2. 创建参与运算的稀疏矩阵A,B,C,D, 此处省略CSR格式矩阵数据的初始化过程
musparseSpMatDescr_t matA, matB, matC, matD;

CHECK_MUSPARSE(musparseCreateCsr(&matA, M, K, nnz_A, d_row_ptr_A, d_col_idx_A, d_csr_values_A,
                                 MUSPARSE_INDEX_32I, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO,
                                 computeType),
               "create matA");
CHECK_MUSPARSE(musparseCreateCsr(&matB, K, N, nnz_B, d_row_ptr_B, d_col_idx_B, d_csr_values_B,
                                 MUSPARSE_INDEX_32I, MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO,
                                 computeType),
               "create matB");
CHECK_MUSPARSE(musparseCreateCsr(&matC, M, N, 0, *d_row_ptr_C, nullptr, nullptr, MUSPARSE_INDEX_32I,
                                 MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO, computeType),
               "create matC");
CHECK_MUSPARSE(musparseCreateCsr(&matD, 0, 0, 0, nullptr, nullptr, nullptr, MUSPARSE_INDEX_32I,
                                 MUSPARSE_INDEX_32I, MUSPARSE_INDEX_BASE_ZERO, computeType),
               "create matC");

mdouble alpha = 1.0f;
mdouble beta = 0.0f;
musparseOperation_t trans_A = MUSPARSE_OPERATION_NON_TRANSPOSE;
musparseOperation_t trans_B = MUSPARSE_OPERATION_NON_TRANSPOSE;

musparseSetPointerMode(handle, MUSPARSE_POINTER_MODE_HOST);

// 3.计算buffer_size
size_t buffer_size;
void *dbuffer = nullptr;

// buffer size
musparseSpGEMM(handle, trans_A, trans_B, &alpha, matA, matB, &beta, matC, matC, computeType, alg,
               MUSPARSE_SPGEMM_STAGE_BUFFER_SIZE, &buffer_size, dbuffer);

// allocate buffer
musaMalloc(&dbuffer, buffer_size);

// calculate nnzC
musparseSpGEMM(handle, trans_A, trans_B, &alpha, matA, matB, &beta, matC, matC, computeType, alg,
               MUSPARSE_SPGEMM_STAGE_NNZ, &buffer_size, dbuffer);

// 4.为结果矩阵分配资源
int64_t C_m, C_n, C_nnz;
CHECK_MUSPARSE(musparseSpMatGetSize(matC, &C_m, &C_n, &C_nnz));
CHECK_ERROR(musaMalloc((void **)d_col_idx_C, C_nnz * sizeof(mint)));
CHECK_ERROR(musaMalloc((void **)d_csr_values_C, C_nnz * sizeof(mdouble)));
*nnz_C = C_nnz;
CHECK_MUSPARSE(musparseCsrSetPointers(matC, *d_row_ptr_C, *d_col_idx_C, *d_csr_values_C));

// 5.执行SpGEMM计算
musparseSpGEMM(handle, trans_A, trans_B, &alpha, matA, matB, &beta, matC, matC, computeType, alg,
               MUSPARSE_SPGEMM_STAGE_COMPUTE, &buffer_size, dbuffer);

// 6.释放资源
CHECK_MUSPARSE(musparseDestroySpMat(matA));
CHECK_MUSPARSE(musparseDestroySpMat(matB));
CHECK_MUSPARSE(musparseDestroySpMat(matC));
CHECK_MUSPARSE(musparseDestroy(handle));
CHECK_ERROR(musaFree(dbuffer));
