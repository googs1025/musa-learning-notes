// ============================================================================
//  示例片段: muSPARSE ILU 分解调用流程
//
//  学习目标:
//    1. 创建稀疏矩阵描述符, 并为 L/U 设置上下三角属性。
//    2. 查询分析/分解所需 buffer, 再执行 ILU 分解。
//    3. 理解 ILU 常用于迭代线性求解器的预条件步骤。
//
//  注意:
//    这个文件是 API 片段, 省略了矩阵数据、变量声明和资源释放, 不能直接单独编译。
//    阅读重点是 muSPARSE 调用顺序。
// ============================================================================
// 创建muSPARSE句柄
musparseHandle_t handle;
musparseCreate(&handle);

// 创建矩阵M的描述信息
musparseMatDescr_t descr_M;
musparseCreateMatDescr(&descr_M);

// 创建矩阵L的描述信息
musparseMatDescr_t descr_L;
musparseCreateMatDescr(&descr_L);
musparseSetMatFillMode(descr_L, MUSPARSE_FILL_MODE_LOWER);
musparseSetMatDiagType(descr_L, MUSPARSE_DIAG_TYPE_UNIT);

// 创建矩阵U的描述信息
musparseMatDescr_t descr_U;
musparseCreateMatDescr(&descr_U);
musparseSetMatFillMode(descr_U, MUSPARSE_FILL_MODE_UPPER);
musparseSetMatDiagType(descr_U, MUSPARSE_DIAG_TYPE_NON_UNIT);

// Create matrix info structure
// 创建矩阵描述信息
musparseMatInfo_t info;
musparseCreateMatInfo(&info);

// 获取所需要的缓存大小
size_t buffer_size_M;
size_t buffer_size_L;
size_t buffer_size_U;

musparseDcsrilu0_bufferSize(handle, m, nnz, descr_M, csr_val, csr_row_ptr, csr_col_ind, info,
                            &buffer_size_M);

musparseDcsrsv_bufferSize(handle, MUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, descr_L, csr_val,
                          csr_row_ptr, csr_col_ind, info, &buffer_size_L);

musparseDcsrsv_bufferSize(handle, MUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, descr_U, csr_val,
                          csr_row_ptr, csr_col_ind, info, &buffer_size_U);

size_t buffer_size = max(buffer_size_M, max(buffer_size_L, buffer_size_U));

// 分配临时缓存
void *temp_buffer;
musaMalloc(&temp_buffer, buffer_size);

// 分析矩阵
musparseDcsrilu0_analysis(handle, m, nnz, descr_M, csr_val, csr_row_ptr, csr_col_ind, info,
                          MUSPARSE_ANALYSIS_POLICY_REUSE, MUSPARSE_SOLVE_POLICY_USE_LEVEL,
                          temp_buffer);

musparseDcsrsv_analysis(handle, MUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, descr_L, csr_val,
                        csr_row_ptr, csr_col_ind, info, MUSPARSE_ANALYSIS_POLICY_REUSE,
                        MUSPARSE_SOLVE_POLICY_USE_LEVEL, temp_buffer);

musparseDcsrsv_analysis(handle, MUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, descr_U, csr_val,
                        csr_row_ptr, csr_col_ind, info, MUSPARSE_ANALYSIS_POLICY_REUSE,
                        MUSPARSE_SOLVE_POLICY_USE_LEVEL, temp_buffer);

// 检测零主元
muInt position;
if (MUSPARSE_EXPORT MUSPARSE_STATUS_ZERO_PIVOT ==
    musparseXcsrilu0_zeroPivot(handle, info, &position))

{
    printf("A has structural zero at A(%d,%d)\n", position, position);
}

// 进行不完整LU分解
musparseDcsrilu0(handle, m, nnz, descr_M, csr_val, csr_row_ptr, csr_col_ind, info,
                 MUSPARSE_SOLVE_POLICY_USE_LEVEL, temp_buffer);

// 检测零主元
if (MUSPARSE_EXPORT MUSPARSE_STATUS_ZERO_PIVOT ==
    musparseXcsrilu0_zeroPivot(handle, info, &position)) {
    printf("U has structural and/or numerical zero at U(%d,%d)\n", position, position);
}

// 求解 Lz = x
musparseDcsrsv_solve(handle, MUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, &alpha, descr_L, csr_val,
                     csr_row_ptr, csr_col_ind, info, x, z, MUSPARSE_SOLVE_POLICY_USE_LEVEL,
                     temp_buffer);

// 求解 Uy = z
musparseDcsrsv_solve(handle, MUSPARSE_OPERATION_NON_TRANSPOSE, m, nnz, &alpha, descr_U, csr_val,
                     csr_row_ptr, csr_col_ind, info, z, y, MUSPARSE_SOLVE_POLICY_USE_LEVEL,
                     temp_buffer);

// 释放资源
musaFree(temp_buffer);
musparseDestroyMatInfo(info);
musparseDestroyMatDescr(descr_M);
musparseDestroyMatDescr(descr_L);
musparseDestroyMatDescr(descr_U);
musparseDestroy(handle);
