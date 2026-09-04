# CUDA Freshman 对照区

这个目录用于放 CUDA 学习对照材料。它和 `code/weekX/` 的 MUSA 主线分开维护。

## 使用规则

1. 这里可以放 CUDA 版本的小例子、阅读笔记和迁移记录。
2. MUSA 改写版优先放回对应的 `code/weekX/` 或 `code/weekX/external-cases/`。
3. 外部仓库源码不要直接批量复制；如果某个仓库 license 不清楚，只引用链接和主题，然后重新实现。
4. 每个案例都要说明来源链接、学习点、迁移状态。

## 主要来源

- Tony-Tan/CUDA_Freshman: <https://github.com/Tony-Tan/CUDA_Freshman>

本仓库当前采用“参考主题，重写实现”的方式使用它。

## 推荐阅读顺序

| 本仓库周次 | CUDA_Freshman 目录 | 学习重点 |
|---|---|---|
| Week 1 | `0_hello_world`, `1_check_dimension`, `2_grid_block`, `5_thread_index`, `7_device_information` | kernel 启动、线程层级、设备查询 |
| Week 2 | `3_sum_arrays`, `4_sum_arrays_timer`, `15_pine_memory`, `17_UVA`, `30_stream`, `34_stream_dependence`, `37_asyncAPI`, `38_stream_call_back` | vector add、计时、pinned、stream、event |
| Week 3 | `8_divergence`, `10_reduceInteger`, `12_reduce_unrolling`, `28_shfl_test`, `29_reduce_shfl` | divergence、reduce、unroll、shuffle |
| Week 4 | `18_sum_array_offset`, `19_AoS`, `20_SoA`, `21_sum_array_offset_unrolling`, `22_transform_matrix2D` | 合并访存、数据布局、转置 |
| Week 5 | `24_shared_memory_read_data`, `25_reduce_integer_shared_memory`, `26_transform_shared_memory`, `27_stencil_1d_constant_read_only` | shared memory、constant memory |

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
