# Week 5 · Shared / Constant / GEMM

对应官方指南 Ch5 + Ch7 + Ch9。用 shared memory、constant memory 和 GEMM 建立性能优化主线。

| 文件 | 主题 |
|---|---|
| `01_shared_basics.mu` | 静态 / 动态 shared memory |
| `02_reduce_shared.mu` | shared reduction |
| `03_transpose_shared.mu` | shared transpose |
| `04_stencil_constant.mu` | constant memory + stencil |
| `05_naive_gemm.mu` | naive GEMM |
| `06_tiled_gemm.mu` | tiled GEMM |
| `07_mublas_sgemm.mu` | muBLAS SGEMM 对比骨架 |

`07_mublas_sgemm` 需要 SDK 提供 muBLAS 头文件和库，使用 `make optional` 构建。
