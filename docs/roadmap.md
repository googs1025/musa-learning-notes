# 6 周学习路线

每周绑定官方指南章节，目标是从环境搭建走到能写算子、跑模型、排查疑难。
官方指南：<https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/>

---

## Week 1 · 概念 + 环境（官方 Ch1–4）

- **官方章节**：Ch1 GPU 并行计算 / Ch2 硬件架构 / Ch3 软件架构 / Ch4 环境搭建
- **目标**：建立完整心智模型，跑通 Hello World
- **学**：
  - SIMT 与 CPU 多线程的区别
  - 曲院架构 SM / 存储层级
  - Toolkit / Runtime / Driver 分层
- **做**：
  - 按官方"安装指导"装 Driver + DDK + MUSA Toolkit（或 AutoDL 摩尔线程实例）
  - 跑通官方 "入门指南" 的 Hello World 与 vectorAdd
  - `mcc --version`、设备信息查询命令验证环境
- **产出**：`code/week1/` 跑通的 Hello World + 一份 CUDA → MUSA 命名速查表

---

## Week 2 · 编程基础（官方 Ch5）

- **官方章节**：Ch5 MUSA 编程基础
- **目标**：理解 kernel 编写全流程，掌握 Runtime API 与 Driver API 的取舍
- **学**：
  - `__global__` / `__device__` / `__host__` 修饰符
  - Grid / Block / Thread 索引与 1D→3D 映射
  - `musaMemcpy` H2D / D2H / D2D
  - Stream、Event、MUSA Graph
- **做**：
  - Runtime API 写 vectorAdd
  - Driver API 重写一遍（module / function / launch）
  - Multi-Stream 三阶段流水线
  - vectorAdd 改 MUSA Graph，量化 launch 开销
- **产出**：`code/week2/` 双 API + 多 Stream + Graph 四个示例

---

## Week 3 · 内存层级 & GEMM（官方 Ch5 + Ch9 部分）

- **官方章节**：Ch5（存储相关）+ Ch9 性能优化
- **目标**：理解多级存储，写出 Tiled GEMM
- **学**：
  - Register / Shared / Global / Constant / Texture 延迟与作用域
  - Coalesced Access、Bank Conflict 入门
- **做**：
  - 朴素 GEMM（global only）
  - Tiled GEMM（`__shared__` + `__syncthreads()`）
  - 测吞吐（GFLOPS），与 muBLAS 对比
- **产出**：性能数据表 + 笔记《GEMM 的三阶进化》

---

## Week 4 · 性能优化（官方 Ch9）

- **官方章节**：Ch9 MUSA 性能优化
- **目标**：会用官方 profiler 找瓶颈
- **学**：
  - Warp Divergence、Bank Conflict、Memory Coalescing 实战
  - Occupancy 与 block size 选择
  - 官方 profiler 工具（参考 Ch9 与"入门指南"）
- **做**：
  - 写一个有 / 无 warp divergence 的 kernel，对比耗时
  - 矩阵转置 with / without padding 验证 bank conflict
  - 用 profiler 分析 Week 3 的 Tiled GEMM，找下一步优化点
- **产出**：profiler 截图 + 优化前后对比表

---

## Week 5 · 加速库 & 框架（官方 Ch7 + Ch8 + Ch10）

- **官方章节**：Ch7 数学加速库 / Ch8 MCCL / Ch10 AI 框架算子
- **目标**：从写 kernel 转向"用库 + 调框架"
- **学**：
  - muBLAS / muDNN / muFFT 接口风格
  - MCCL 通信原语（AllReduce / Broadcast / ReduceScatter）
  - torch_musa 自定义算子注册流程
- **做**：
  - muBLAS SGEMM vs Week 3 自写 GEMM
  - torch_musa 跑 ResNet 或小 LLM 推理
  - MCCL 双卡 AllReduce 测带宽（如有 2 卡环境）
  - 写一个 torch_musa Custom Op
- **产出**："自写 vs 加速库 vs torch_musa" 对比表

---

## Week 6 · 调试 & 集群视角（官方 Ch6 + Ch11）

- **官方章节**：Ch6 MUSA 调试 / Ch11 附录（错误码表）
- **目标**：能定位疑难 bug，能讲清 KUAE 集群方案
- **学**：
  - MUSA GDB、Error Dump 流程
  - 常见错误码（`MUSA_ERROR_*`）含义
  - KUAE Fusion / Matrix + MTLink 拓扑（与 NV DGX SuperPOD 对照）
- **做**：
  - 故意越界 kernel 触发 `MUSA_ERROR_ILLEGAL_ADDRESS`，按官方流程解析
  - MUSA GDB 单步 kernel
  - 调研 K8s 上 MUSA 设备发现 / 调度方案
- **产出**：`notes/troubleshooting.md` 故障排查手册 v0.1
