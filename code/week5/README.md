# Week 5 · Shared / Constant / GEMM

对应官方指南 Ch5 + Ch7 + Ch9。用 shared memory、constant memory 和 GEMM 建立性能优化主线。

本周教材见 [`learning-notes.md`](learning-notes.md)：从 shared memory 基础一路读到 tiled GEMM 和 muBLAS 对比。

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

CUDA_Freshman 对照案例和迁移计划见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。

## CUDA reference 精选

本周另有一组保留 CUDA 写法的 `.cu` 参考，位于 `cuda-reference/`。它们固定来自
[kriegalex/wrox-pro-cuda-c](https://github.com/kriegalex/wrox-pro-cuda-c)
commit `63825d64683b644198dd9cb0d4d472d6914d4f72`，可用同一 Makefile 在
CUDA/MUSA 后端之间切换：

~~~bash
cd code/week5/cuda-reference
make BACKEND=cuda
make BACKEND=musa MUSA_ARCH=mp_31
make BACKEND=cuda TARGET=chapter07__fmad
~~~

参考目录覆盖 shared/constant/reduce、atomic 和浮点行为。默认构建排除
`chapter05__reduceIntegerShfl`，因为它有 CUDA 32-lane shuffle 假设；该目标
可显式构建，但 MUSA 支持必须单独验证。没有 SDK 时只能做 dry-run，不能把
命令生成当成编译或性能结论。
