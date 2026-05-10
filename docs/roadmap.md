# 6 周学习路线 · 细化版

每周绑定官方[编程指南](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/)章节。
示例颗粒度参考 [Tony-Tan/CUDA_Freshman](https://github.com/Tony-Tan/CUDA_Freshman):一个示例只演示一个点,加 timer / 改数据布局 / 加 unroll 都拆成独立的小例子。

> 完成度图例:✅ 已发布   🧪 写好待跑   ⏳ 计划中

---

## Week 1 · 入门 + 软件栈(官方 Ch1–4) ✅

- **学**:SIMT 模型 / 曲院架构 / Toolkit 三层栈 / `__global__` / `<<<>>>` / `dim3`
- **做**:6 个最小示例,全部 ✅

| # | 文件 | 主题 |
|---|---|---|
| 01 | `01_hello_world.mu` | GPU printf,理解 kernel 启动语法 |
| 02 | `02_thread_index.mu` | `threadIdx` / `blockIdx` / `blockDim` |
| 03 | `03_device_info.mu` | `musaDeviceProp` 设备查询 |
| 04 | `04_memory_basics.mu` | 显存四件套:Malloc / Memset / Memcpy / Free |
| 05 | `05_error_check.mu` | 同步 vs 异步错误,CHECK 范式 |
| 06 | `06_async_kernel.mu` | kernel 异步语义 + launch overhead |

**产出**:6 示例 + 10 道习题 + 公众号文章 [`docs/articles/01-first-musa-code.md`](articles/01-first-musa-code.md)

---

## Week 2 · 编程模型 + Stream + Graph(官方 Ch5)⏳

- **学**:GPU 程序 7 步骨架 / Runtime vs Driver API / Stream 并发 / Event 计时 / Graph 重放
- **做**:8 个示例,粒度参考 Freshman 30–38(Stream 系列)

| # | 文件 | 主题 |
|---|---|---|
| 01 | `01_vector_add_runtime.mu` | Runtime API 完整流程 |
| 02 | `02_vector_add_driver.mu` | Driver API 重写,`muLaunchKernel` |
| 03 | `03_vector_add_timer.mu` | `musaEvent` 计时 vs CPU 计时 |
| 04 | `04_vector_add_unified.mu` | `musaMallocManaged` 统一内存版 |
| 05 | `05_multi_stream.mu` | 多流流水线:H2D / Kernel / D2H 三阶段重叠 |
| 06 | `06_stream_event_dep.mu` | 跨流通过 event 同步 |
| 07 | `07_musa_graph.mu` | Stream Capture → Graph,量化 launch overhead |
| 08 | `08_stream_callback.mu` | 在 stream 完成时回调 host 函数 |

**产出**:8 示例 + 文章《Stream 与 Graph 把吞吐量挤到极限》

---

## Week 3 · 执行模型(官方 Ch5 + Freshman Ch3)⏳

- **学**:Warp Divergence / 循环展开 / Warp Shuffle / 动态并行 / 多维 grid
- **做**:6 个示例,Reduce 是主线

| # | 文件 | 主题 |
|---|---|---|
| 01 | `01_warp_divergence.mu` | 故意制造 if/else 分歧,实测耗时 |
| 02 | `02_reduce_naive.mu` | 朴素归约(global memory) |
| 03 | `03_reduce_unrolling.mu` | 循环展开 + 多元素/线程 |
| 04 | `04_reduce_shfl.mu` | warp shuffle 内归约(MUSA warp = 128) |
| 05 | `05_nested_hello.mu` | 动态并行:kernel 启动 kernel |
| 06 | `06_sum_matrix_2d.mu` | 多维 grid,2D 矩阵求和 |

**产出**:6 示例 + Reduce 三阶进化对比表

---

## Week 4 · 全局内存与访存(官方 Ch9 + Freshman Ch4)⏳

- **学**:Coalesced Access / SoA vs AoS / Bank Conflict / Transpose
- **做**:6 个示例,主题就是"访存模式怎么决定吞吐"

| # | 文件 | 主题 |
|---|---|---|
| 01 | `01_saxpy_bandwidth.mu` | 测理论带宽利用率 |
| 02 | `02_offset_access.mu` | offset 0/1/.../31 对吞吐的影响 |
| 03 | `03_offset_unrolling.mu` | 偏移 + 展开,看哪个更值 |
| 04 | `04_aos_vs_soa.mu` | AoS / SoA 两版数据布局对比 |
| 05 | `05_transpose_naive.mu` | 朴素转置,bank conflict 现场 |
| 06 | `06_transpose_padded.mu` | 用 padding 消 bank conflict |

**产出**:6 示例 + 文章《访存模式才是 GPU 吞吐的天花板》

---

## Week 5 · Shared / Constant / GEMM(官方 Ch5 + Freshman Ch5)⏳

- **学**:Shared Memory / Constant Memory / Tiled GEMM / 加速库
- **做**:7 个示例,用 GEMM 串起整章

| # | 文件 | 主题 |
|---|---|---|
| 01 | `01_shared_basics.mu` | 静态 / 动态 shared memory 用法 |
| 02 | `02_reduce_shared.mu` | 用 shared 重写 reduce,看带宽提升 |
| 03 | `03_transpose_shared.mu` | shared memory 解决转置带宽 |
| 04 | `04_stencil_constant.mu` | constant memory + 1D stencil |
| 05 | `05_naive_gemm.mu` | 朴素 GEMM(global only) |
| 06 | `06_tiled_gemm.mu` | Tiled GEMM:`__shared__` + `__syncthreads()` |
| 07 | `07_mublas_sgemm.mu` | muBLAS SGEMM,跟自写版对比 |

**产出**:7 示例 + GFLOPS 对比表 + 笔记《GEMM 三阶进化》

---

## Week 6 · 多卡 + 调试 + 框架(官方 Ch6 + Ch8 + Ch10)⏳

- **学**:MUSA GDB / 错误码 / MCCL 通信 / torch_musa 自定义算子 / 集群视角(KUAE)
- **做**:5 个示例,MUSA 特色路径(Freshman 这章不覆盖)

| # | 文件 | 主题 |
|---|---|---|
| 01 | `01_mccl_allreduce.mu` | 双卡 AllReduce(如有 2 卡环境) |
| 02 | `02_musa_gdb_demo.mu` | 故意 illegal address,用 mcc-gdb 单步 |
| 03 | `03_error_dump.mu` | 解析 MUSA Error Dump 流程 |
| 04 | `04_torch_musa_minimal.py` | torch_musa 跑 ResNet 推理 |
| 05 | `05_torch_musa_custom_op.cpp` | 注册一个 torch_musa Custom Op |

**产出**:5 示例 + 故障排查手册 v0.1

---

## 总览

```
Week 1 (Ch1-4)   ✅  6 示例  ── 入门
Week 2 (Ch5)     ⏳  8 示例  ── Stream / Graph
Week 3 (Ch5+9)   ⏳  6 示例  ── Warp / Reduce
Week 4 (Ch9)     ⏳  6 示例  ── 访存模式
Week 5 (Ch5+9)   ⏳  7 示例  ── Shared / GEMM
Week 6 (Ch6+8+10)⏳  5 示例  ── 调试 / 多卡 / 框架
─────────────────────────────────
                    38 示例(对标 Freshman 39)
```

> 节奏不强求 6 周完成,有空就推一周。每完成一周补一篇笔记到 `docs/articles/`。

---

## 学习参考

- **官方编程指南**:<https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/>
- **CUDA C++ 编程指南**:<https://docs.nvidia.com/cuda/cuda-c-programming-guide/>(API 几乎可对应,先吃 CUDA 文档,再看 MUSA 差异)
- **CUDA_Freshman**:<https://github.com/Tony-Tan/CUDA_Freshman>(本路线图示例颗粒度的参考)
- **专业丛书**:《CUDA C 编程权威指南》《Programming Massively Parallel Processors》