# CUDA Example Map

这个文档用来管理外部 CUDA 代码例子和本仓库 MUSA 学习路线的对应关系。

原则：

1. CUDA 原项目用于学习主题和代码组织，不直接批量复制源码。
2. 如果要保留 CUDA 代码，放在 `code/cuda-freshman/` 等独立目录，不能混进 `code/weekX/` 主线。
3. MUSA 改写版放在对应 `code/weekX/` 或 `code/weekX/external-cases/`。
4. 每个迁移案例都要写清楚来源、学习点、CUDA 到 MUSA 的改写点和运行观察。

## 来源

- Tony-Tan/CUDA_Freshman: <https://github.com/Tony-Tan/CUDA_Freshman>
- NVIDIA cuda-samples: <https://github.com/NVIDIA/cuda-samples>
- SGEMM_CUDA: <https://github.com/siboehm/SGEMM_CUDA>

Tony-Tan/CUDA_Freshman 的根目录没有看到明确 LICENSE 文件，因此本仓库默认只引用主题和链接，MUSA 代码采用重新实现。

## Week 1

| CUDA_Freshman 目录 | 对应主题 | 本仓库位置 | 状态 |
|---|---|---|---|
| `0_hello_world` | kernel 启动、GPU printf | `code/week1/01_hello_world.mu` | 已有 MUSA 主线 |
| `1_check_dimension` | grid/block 维度检查 | `code/week1/02_thread_index.mu` | 已覆盖核心概念 |
| `2_grid_block` | launch config 与线程层级 | `code/week1/02_thread_index.mu` | 已覆盖核心概念 |
| `5_thread_index` | 一维/二维索引 | `code/week1/02_thread_index.mu` | 已覆盖核心概念 |
| `7_device_information` | device query | `code/week1/03_device_info.mu` | 已有 MUSA 主线 |

可补案例：

- `external-cases/01_check_dimension_notes.md`: 用 CUDA_Freshman 的维度检查案例补一篇对照笔记，已完成。
- `external-cases/01_check_dimension_musa.mu`: 重写一个只打印 `gridDim/blockDim` 的极简 MUSA 版本，已完成。
- `external-cases/02_grid_block_musa.mu`: 重写一个只打印 `gridDim/blockDim` 的极简 MUSA 版本。

## Week 2

| CUDA_Freshman 目录 | 对应主题 | 本仓库位置 | 状态 |
|---|---|---|---|
| `3_sum_arrays` | vector add 基础流程 | `code/week2/01_vector_add_runtime.mu` | 已有 MUSA 主线 |
| `4_sum_arrays_timer` | CPU timer 与 GPU timer | `code/week2/03_vector_add_timer.mu` | 已有 MUSA 主线 |
| `15_pine_memory` | pinned memory | `code/week2/02_vector_add_pinned.mu` | 已有 MUSA 主线 |
| `16_zero_copy_memory` | zero-copy memory | `code/week2/external-cases/` | 待评估 MUSA 支持 |
| `17_UVA` | unified virtual addressing | `code/week2/04_vector_add_unified.mu` | 部分覆盖 |
| `30_stream` | stream 基础 | `code/week2/05_multi_stream.mu` | 已有 MUSA 主线 |
| `34_stream_dependence` | stream/event 依赖 | `code/week2/06_stream_event_dep.mu` | 已有 MUSA 主线 |
| `37_asyncAPI` | async copy/API | `code/week2/external-cases/` | 待迁移 |
| `38_stream_call_back` | stream callback | `code/week2/08_stream_callback.mu` | 已有 MUSA 主线 |

可补案例：

- `external-cases/01_zero_copy_probe.mu`: 探测 MUSA 是否支持 CUDA 风格 zero-copy。
- `external-cases/02_async_copy_musa.mu`: 用 `musaMemcpyAsync` 对比同步拷贝。

## Week 3

| CUDA_Freshman 目录 | 对应主题 | 本仓库位置 | 状态 |
|---|---|---|---|
| `8_divergence` | warp divergence | `code/week3/01_warp_divergence.mu` | 已有 MUSA 主线 |
| `9_sum_matrix2D` | 2D grid 矩阵处理 | `code/week3/06_sum_matrix_2d.mu` | 已有 MUSA 主线 |
| `10_reduceInteger` | naive reduce | `code/week3/02_reduce_naive.mu` | 已有 MUSA 主线 |
| `11_simple_sum_matrix2D` | 2D 矩阵求和 | `code/week3/06_sum_matrix_2d.mu` | 已覆盖核心概念 |
| `12_reduce_unrolling` | reduce unrolling | `code/week3/03_reduce_unrolling.mu` | 已有 MUSA 主线 |
| `13_nested_hello_world` | dynamic parallelism | `code/week3/05_nested_hello.mu` | 已有 MUSA 主线 |
| `28_shfl_test` | shuffle API 探测 | `code/week3/external-cases/` | 待迁移 |
| `29_reduce_shfl` | shuffle reduce | `code/week3/04_reduce_shfl.mu` | 已有 MUSA 主线 |

可补案例：

- `external-cases/01_reduce_from_cuda_freshman.md`: 对照 naive/unroll/shuffle 三种 reduce。
- `external-cases/02_shuffle_probe.mu`: 单独验证 MUSA shuffle intrinsic 和 warp size。

## Week 4

| CUDA_Freshman 目录 | 对应主题 | 本仓库位置 | 状态 |
|---|---|---|---|
| `18_sum_array_offset` | offset 破坏合并访存 | `code/week4/02_offset_access.mu` | 已有 MUSA 主线 |
| `19_AoS` | Array of Structs | `code/week4/04_aos_vs_soa.mu` | 已有 MUSA 主线 |
| `20_SoA` | Struct of Arrays | `code/week4/04_aos_vs_soa.mu` | 已有 MUSA 主线 |
| `21_sum_array_offset_unrolling` | offset + unrolling | `code/week4/03_offset_unrolling.mu` | 已有 MUSA 主线 |
| `22_transform_matrix2D` | matrix transpose | `code/week4/05_transpose_naive.mu` | 已覆盖 naive 版本 |

可补案例：

- `external-cases/01_coalescing_checklist.md`: 把 offset/AoS/SoA/transpose 串成访存诊断表。
- `external-cases/02_transpose_cuda_to_musa.md`: 对照 CUDA_Freshman 转置和 MUSA shared/padded 版本。

## Week 5

| CUDA_Freshman 目录 | 对应主题 | 本仓库位置 | 状态 |
|---|---|---|---|
| `14_global_variable` | global/constant symbol | `code/week5/04_stencil_constant.mu` | 部分覆盖 |
| `24_shared_memory_read_data` | shared memory 读写 | `code/week5/01_shared_basics.mu` | 已有 MUSA 主线 |
| `25_reduce_integer_shared_memory` | shared reduce | `code/week5/02_reduce_shared.mu` | 已有 MUSA 主线 |
| `26_transform_shared_memory` | shared transpose | `code/week5/03_transpose_shared.mu` | 已有 MUSA 主线 |
| `27_stencil_1d_constant_read_only` | constant/read-only cache | `code/week5/04_stencil_constant.mu` | 已有 MUSA 主线 |

可补案例：

- `external-cases/01_shared_reduce_from_cuda.md`: 解释 global reduce 到 shared reduce 的收益。
- `external-cases/02_sgemm_progression.md`: 引入 SGEMM_CUDA 的 naive 到 tiled 优化路线。

## Week 6

CUDA_Freshman 基本不覆盖 MUSA 调试、多卡、MCCL 和 torch_musa。本周外部代码来源应优先看：

- MUSA SDK 官方示例
- MCCL 官方示例
- torch_musa 官方文档和示例
- NVIDIA cuda-samples 中的 `simpleMultiGPU`、debugging 和 profiler 类示例，仅做概念对照

可补案例：

- `external-cases/01_multigpu_cuda_vs_mccl.md`: 解释 CUDA 多卡例子迁移到 MCCL/MUSA 时哪些概念保留。
- `external-cases/02_debugging_workflow.md`: 对照 CUDA 调试流程和 MUSA GDB/Error Dump。
