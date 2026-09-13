# CUDA Example Map

这个文档用来管理外部 CUDA 代码例子和本仓库 MUSA 学习路线的对应关系。

原则：

1. CUDA 原项目用于学习主题和代码组织，不直接批量复制源码。
2. 保留 CUDA 原码时放在 `code/weekX/cuda-reference/` 等明确隔离的 reference 目录，不能混进 `code/weekX/` 的 MUSA 主线。
3. MUSA 改写版放在对应 `code/weekX/` 或 `code/weekX/external-cases/`。
4. 每个迁移案例都要写清楚来源、学习点、CUDA 到 MUSA 的改写点和运行观察。

## 来源

- Tony-Tan/CUDA_Freshman: <https://github.com/Tony-Tan/CUDA_Freshman>
- NVIDIA cuda-samples: <https://github.com/NVIDIA/cuda-samples>
- SGEMM_CUDA: <https://github.com/siboehm/SGEMM_CUDA>
- alexngng/CUDA-Learn-Note: <https://github.com/alexngng/CUDA-Learn-Note>
- kriegalex/wrox-pro-cuda-c: <https://github.com/kriegalex/wrox-pro-cuda-c>

`wrox-pro-cuda-c` 本次使用固定 commit `63825d64683b644198dd9cb0d4d472d6914d4f72`，保留上游 MIT 版权与 `.cu`/CUDA API 语义，精选文件按 Week 放入 `code/weekN/cuda-reference/`。目录内 Makefile 用 `BACKEND=cuda` 走 `nvcc`，用 `BACKEND=musa` 走 `mcc -mtgpu`；MUSA 状态表示兼容性待 SDK/设备实测，不代表已编译通过。实际构建检查使用 `make -n BACKEND=cuda` 或 `make -n BACKEND=musa MUSA_ARCH=mp_31`；没有对应 SDK 时不把 dry-run 记录为编译通过。

`CUDA-Learn-Note` 更适合作为 kernel 复习和性能思路参考，而不是本仓库的逐例迁移源。它覆盖 SGEMM/SGEMV、warp/block reduce、dot product、elementwise、histogram、softmax、LayerNorm/RMSNorm 等小型 kernel，适合按 Week 3–5 的主题交叉阅读。阅读时要特别检查其中对 CUDA warp size、shuffle intrinsic、向量化类型和同步语义的假设，再决定如何映射到 MUSA。

Tony-Tan/CUDA_Freshman 的根目录没有看到明确 LICENSE 文件，因此本仓库默认只引用主题和链接，MUSA 代码采用重新实现。

## wrox-pro-cuda-c 已导入文件

### Week 1：执行模型与索引（Ch1–Ch2）

| 来源 chapter / 文件 | 主题 | 现有对应主线 | 状态 |
|---|---|---|---|
| `chapter01/hello.cu` | kernel 启动、GPU printf | `code/week1/01_hello_world.mu` | CUDA reference；MUSA 未实编 |
| `chapter02/checkDeviceInfor.cu` | 设备属性查询 | `code/week1/03_device_info.mu` | CUDA reference；属性字段依赖 SDK |
| `chapter02/checkDimension.cu` | grid/block/thread 维度 | `code/week1/02_thread_index.mu` | CUDA reference；MUSA 未实编 |
| `chapter02/checkThreadIndex.cu` | 全局线程索引 | `code/week1/02_thread_index.mu` | CUDA reference；MUSA 未实编 |
| `chapter02/defineGridBlock.cu` | launch 配置 | `code/week1/02_thread_index.mu` | CUDA reference；设备上限依赖硬件 |
| `chapter02/sumArraysOnGPU-small-case.cu` | 单 block 向量加 | `code/week1/04_memory_basics.mu` | CUDA reference；MUSA 未实编 |
| `chapter02/sumArraysOnGPU-timer.cu` | CPU/GPU 计时 | `code/week1/06_async_kernel.mu` | CUDA reference；数字不可跨设备外推 |
| `chapter02/sumMatrixOnGPU-1D-grid-1D-block.cu` | 1D grid/1D block 矩阵加 | `code/week3/06_sum_matrix_1d.mu` | CUDA reference；MUSA 未实编 |
| `chapter02/sumMatrixOnGPU-2D-grid-1D-block.cu` | 2D grid/1D block 矩阵加 | `code/week3/06_sum_matrix_2d.mu` | CUDA reference；MUSA 未实编 |
| `chapter02/sumMatrixOnGPU-2D-grid-2D-block.cu` | 2D grid/2D block 矩阵加 | `code/week3/06_sum_matrix_2d.mu` | CUDA reference；MUSA 未实编 |

### Week 2：内存与异步（Ch4、Ch6）

| 来源 chapter / 文件 | 主题 | 现有对应主线 | 状态 |
|---|---|---|---|
| `chapter04/memTransfer.cu` | pageable 显式 H2D/D2H | `code/week2/01_vector_add_runtime.mu` | CUDA reference；MUSA 未实编 |
| `chapter04/pinMemTransfer.cu` | pinned host memory | `code/week2/02_vector_add_pinned.mu` | CUDA reference；需 pinned 支持 |
| `chapter04/sumArrayZerocpy.cu` | zero-copy mapped memory | `code/week2/04_vector_add_unified.mu` | CUDA/硬件依赖；MUSA 不承诺 |
| `chapter04/sumMatrixGPUManaged.cu` | managed memory | `code/week2/04_vector_add_unified.mu` | CUDA reference；迁移语义待实测 |
| `chapter04/sumMatrixGPUManual.cu` | 显式矩阵拷贝基线 | `code/week2/01_vector_add_runtime.mu` | CUDA reference；MUSA 未实编 |
| `chapter06/asyncAPI.cu` | async copy、event | `code/week2/05_multi_stream.mu` | CUDA reference；需 pinned buffer |
| `chapter06/simpleCallback.cu` | stream callback | `code/week2/08_stream_callback.mu` | API/回调线程语义待实测 |
| `chapter06/simpleHyperqBreadth.cu` | Hyper-Q breadth submission | `code/week2/05_multi_stream.mu` | CUDA/特定硬件观察例 |
| `chapter06/simpleHyperqDependence.cu` | 跨 stream event 依赖 | `code/week2/06_stream_event_dep.mu` | CUDA/特定并发能力观察例 |

其余 `chapter03`、`chapter05`、`chapter07`–`chapter10` 不在 Task 2 导入范围，后续任务按 Week 3–6 的主题精选；本次不复制未列出的上游文件。

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

### wrox-pro-cuda-c：Week 6 导入（Ch8–Ch10）

来源固定为 `63825d64683b644198dd9cb0d4d472d6914d4f72`，目标目录为 `code/week6/cuda-reference/`。

| 来源文件 | 主题 | 本仓库位置 | 状态 |
|---|---|---|---|
| `chapter08/cublas.cu` | cuBLAS SGEMV | `code/week6/cuda-reference/chapter08/cublas.cu` | CUDA library；MUSA Mapping 候选，未验证 |
| `chapter08/cusparse.cu` | CSR SpMV | `code/week6/cuda-reference/chapter08/cusparse.cu` | CUDA library；MUSA 对应库未验证 |
| `chapter08/cufft.cu` | C2C FFT | `code/week6/cuda-reference/chapter08/cufft.cu` | CUDA library；MUSA 对应库未验证 |
| `chapter09/simpleMultiGPU.cu` | 多卡异步分片 | `code/week6/cuda-reference/chapter09/simpleMultiGPU.cu` | CUDA 多 GPU；MUSA 多卡未验证 |
| `chapter09/simpleP2P.c` | MPI host staging | `code/week6/cuda-reference/chapter09/simpleP2P.c` | source-only；CUDA + MPI 需独立 `mpicc`/MPI 与两进程两卡；不进入 reference Makefile target |
| `chapter09/simpleP2P_PingPong.cu` | GPU P2P | `code/week6/cuda-reference/chapter09/simpleP2P_PingPong.cu` | CUDA P2P；MUSA 拓扑/API 未验证 |
| `chapter10/debug-hazards.cu` | 共享归约调试风险 | `code/week6/cuda-reference/chapter10/debug-hazards.cu` | CUDA 调试候选，未实编 |
| `chapter10/debug-segfault.cu` | 故意 pointer 错误 | `code/week6/cuda-reference/chapter10/debug-segfault.cu` | CUDA 故障样例；MUSA 未验证 |
| `chapter10/debug-segfault.fixed.cu` | pointer table 修复 | `code/week6/cuda-reference/chapter10/debug-segfault.fixed.cu` | CUDA 对照；MUSA 未验证 |
| `chapter10/sumMatrixGPU.cu` | 2D 矩阵加法 | `code/week6/cuda-reference/chapter10/sumMatrixGPU.cu` | CUDA runtime；MUSA Mapping 候选 |
| `chapter10/crypt.parallelized.cu` | IDEA 并行加密 | `code/week6/cuda-reference/chapter10/crypt.parallelized.cu` | CUDA kernel；输入/密钥依赖 |
| `chapter10/crypt.overlap.cu` | IDEA stream overlap | `code/week6/cuda-reference/chapter10/crypt.overlap.cu` | CUDA streams；MUSA 未验证 |

本次只纳入上述 12 个文件。其余 Ch8–Ch10 文件及未列出的 Ch1–Ch7 文件不纳入；后续加入必须重新登记依赖和验证状态。

### wrox-pro-cuda-c：Week 3 导入（Ch3、Ch5）

来源固定为 `63825d64683b644198dd9cb0d4d472d6914d4f72`，目标目录为 `code/week3/cuda-reference/`。target 名按 `chapterXX__name` 约定生成；动态并行和旧式 shuffle 默认不进入 `all`。

| 来源文件 | 主题 | target | 状态 |
|---|---|---|---|
| `chapter03/simpleDivergence.cu` | warp divergence | `chapter03__simpleDivergence` | CUDA/MUSA 均未实编；warp 行为需实测 |
| `chapter03/reduceInteger.cu` | shared reduction 与 unroll | `chapter03__reduceInteger` | CUDA/MUSA 未实编；block 配置需验证 |
| `chapter03/nestedHelloWorld.cu` | dynamic parallelism | `chapter03__nestedHelloWorld` | optional；CUDA 架构支持与 MUSA 支持均未验证 |
| `chapter03/nestedReduce.cu` | device-side recursive launch | `chapter03__nestedReduce` | optional；不进入默认 `all`，未验证 |
| `chapter03/sumMatrix.cu` | 2D grid 矩阵加法 | `chapter03__sumMatrix` | CUDA/MUSA 未实编 |
| `chapter03/reduceIntegerShfl.cu` | shuffle reduction | `chapter03__reduceIntegerShfl` | CUDA/MUSA 未实编；warp/mask 语义需验证 |
| `chapter03/simpleShfl.cu` | shuffle API 探针 | `chapter03__simpleShfl` | optional；旧 intrinsic 与 32-lane 假设，未验证 |
| `chapter05/checkSmemSquare.cu` | shared shape / bank mapping | `chapter05__checkSmemSquare` | CUDA/MUSA 未实编；设备布局需实测 |
| `chapter05/checkSmemRectangle.cu` | rectangular shared tile | `chapter05__checkSmemRectangle` | CUDA/MUSA 未实编；设备布局需实测 |

### wrox-pro-cuda-c：Week 4 导入（Ch4）

目标目录为 `code/week4/cuda-reference/`。这些文件用于访存布局与转置对照，性能数字必须按设备和后端分别记录。

| 来源文件 | 主题 | target | 状态 |
|---|---|---|---|
| `chapter04/readSegment.cu` | 连续读取与合并访存 | `chapter04__readSegment` | CUDA/MUSA 未实编 |
| `chapter04/writeSegment.cu` | 连续写入与 stride | `chapter04__writeSegment` | CUDA/MUSA 未实编 |
| `chapter04/readSegmentUnroll.cu` | offset + unroll | `chapter04__readSegmentUnroll` | CUDA/MUSA 未实编；性能不推定 |
| `chapter04/simpleMathAoS.cu` | Array of Structs | `chapter04__simpleMathAoS` | CUDA/MUSA 未实编 |
| `chapter04/simpleMathSoA.cu` | Struct of Arrays | `chapter04__simpleMathSoA` | CUDA/MUSA 未实编 |
| `chapter04/transpose.cu` | global/shared 转置 | `chapter04__transpose` | CUDA/MUSA 未实编；bank 行为需实测 |
| `chapter04/globalVariable.cu` | device/global symbol | `chapter04__globalVariable` | CUDA/MUSA 未实编；symbol 映射需验证 |

### wrox-pro-cuda-c：Week 5 导入（Ch5、Ch7）

目标目录为 `code/week5/cuda-reference/`。shared、constant、atomic、浮点和 FMAD 示例均保留 CUDA 参考语义，不能把“可生成编译命令”解释为后端已支持。

| 来源文件 | 主题 | target | 状态 |
|---|---|---|---|
| `chapter05/checkSmemSquare.cu` | shared shape / padding | `chapter05__checkSmemSquare` | CUDA/MUSA 未实编 |
| `chapter05/checkSmemRectangle.cu` | rectangular shared access | `chapter05__checkSmemRectangle` | CUDA/MUSA 未实编 |
| `chapter05/constantReadOnly.cu` | constant 与 read-only stencil | `chapter05__constantReadOnly` | CUDA/MUSA 未实编；cache 语义需验证 |
| `chapter05/constantStencil.cu` | constant memory stencil | `chapter05__constantStencil` | CUDA/MUSA 未实编；symbol 拷贝需验证 |
| `chapter05/reduceInteger.cu` | global/shared/unroll reduction | `chapter05__reduceInteger` | CUDA/MUSA 未实编 |
| `chapter05/reduceIntegerShfl.cu` | shuffle reduction | `chapter05__reduceIntegerShfl` | optional；CUDA 32-lane 假设，MUSA 未验证 |
| `chapter07/my-atomic-add.cu` | CAS custom atomic add | `chapter07__my-atomic-add` | CUDA/MUSA 未实编；atomicCAS 支持需验证 |
| `chapter07/atomic-ordering.cu` | atomic 与更新顺序 | `chapter07__atomic-ordering` | CUDA/MUSA 未实编；不保证固定输出顺序 |
| `chapter07/floating-point-accuracy.cu` | 浮点表示与误差 | `chapter07__floating-point-accuracy` | CUDA/MUSA 未实编；精度模式影响结果 |
| `chapter07/floating-point-perf.cu` | float/double 传输与计算 | `chapter07__floating-point-perf` | CUDA/MUSA 未实编；性能不推定 |
| `chapter07/fmad.cu` | FMAD 融合与舍入 | `chapter07__fmad` | CUDA/MUSA 未实编；依赖编译器与架构 |
