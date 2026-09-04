# Week 2 学习材料

Week 2 把 Week 1 的单个 kernel 扩展成完整 GPU 程序：准备输入、拷贝到 device、执行 kernel、计时、拷回结果，并进一步用 stream/event/graph 组织异步任务。

## 阅读顺序

1. `01_vector_add_runtime.mu`: 固定 Runtime API 的 7 步骨架。
2. `03_vector_add_timer.mu`: 先把计时方法学准。
3. `02_vector_add_pinned.mu`: 再看 pageable 和 pinned host memory 的拷贝差异。
4. `04_vector_add_unified.mu`: 理解统一内存是易用性权衡，不是免费性能。
5. `05_multi_stream.mu`: 看 H2D/kernel/D2H 如何流水线化。
6. `06_stream_event_dep.mu`: 用 event 表达跨流依赖。
7. `07_musa_graph.mu`: 用 graph 重放观察 launch overhead。
8. `08_stream_callback.mu`: 看 GPU 工作完成后如何回调 host。

## 核心知识点

| 示例 | 必须掌握 | 常见误区 |
|---|---|---|
| `01_vector_add_runtime.mu` | host/device 分配、H2D、kernel、D2H、验证、释放 | 跳过小输入验证直接跑大数据 |
| `02_vector_add_pinned.mu` | `musaMallocHost`、DMA、pageable vs pinned | 以为 pinned 总是越多越好 |
| `03_vector_add_timer.mu` | CPU 计时必须同步，GPU event 更适合 kernel 时间 | 把 launch 入队时间当 kernel 时间 |
| `04_vector_add_unified.mu` | `musaMallocManaged`、prefetch 能力探测 | 以为 UM 一定自动更快 |
| `05_multi_stream.mu` | 多流流水线和 chunk 切分 | 乱加 device sync 破坏并发 |
| `06_stream_event_dep.mu` | event record/wait 形成 DAG | 用全设备同步表达局部依赖 |
| `07_musa_graph.mu` | stream capture、instantiate、launch replay | 直接套 CUDA graph 性能直觉 |
| `08_stream_callback.mu` | callback 由完成时机触发 | 以为回调顺序等于提交顺序 |

## 代码阅读抓手

本周所有示例都围绕一个问题：任务是否真的重叠。阅读时重点看：

- host memory 是 pageable 还是 pinned。
- memcpy 是同步版本还是 async 版本。
- kernel 和 memcpy 是否在同一个 stream。
- 是否有 event 或 device sync 把本来能并发的工作串行化。

## CUDA_Freshman 对照

- `3_sum_arrays`: vector add 主流程。
- `4_sum_arrays_timer`: 计时方法。
- `15_pine_memory`: pinned memory。
- `17_UVA`: 统一虚拟地址/统一内存相关概念。
- `30_stream`、`34_stream_dependence`、`37_asyncAPI`、`38_stream_call_back`: stream、event、async、callback。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
