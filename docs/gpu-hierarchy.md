# GPU 结构层次：从 MUSA 的 MPC、MPX、MP 到 CUDA

这篇指南分两层来讲：先用“物流园”建立直觉，再回到严格的硬件组织、编程模型和性能含义。

先约定一条边界：MUSA 与 CUDA 的公开术语并不是严格的一一映射。下面的 CUDA 对照只帮助理解编程角色，不能把两家的物理结构逐层翻译，更不能画等号。

## 先看结论

- 软件层次是 `kernel → grid → block → warp → thread`。
- 硬件层次可以概括为 `GPU → MPC → MPX → MP → 执行单元`。
- 一个完整的 block 驻留在一个 MP 上，不会拆到多个 MP；MP 再把 block 中的线程组织成 warp 来执行。
- 从编程角色看，CUDA SM 最接近 MUSA MP：block 在这里驻留，warp 在这里被调度和执行。但不能把 MPC、MPX、MP 依次严格翻译成 GPC、SM sub-partition、SM，两边的层级位置和资源边界并不相同。

## 一张总图

```text
Host CPU
  │ launch kernel
  ▼
MUSA GPU 前端：接收、排队、分发工作
  │
  ├── MPC × N
  │     ├── MPX × N
  │     │     ├── MP × N
  │     │     │     ├── warp 管理与执行
  │     │     │     ├── FP / INT / SFU / TCE 等执行资源
  │     │     │     └── 本地存储与缓存资源
  │     │     └── ...
  │     └── ...
  └── ...
```

图中的 `× N` 都取决于具体架构，不能当成固定数量。“前端”也只是对接收、排队和分发工作的通用教学说法，不是在断言某个芯片内部存在这个名称的独立模块。

## 用物流园来理解

| 概念 | 物流园类比 | 真实含义 |
|---|---|---|
| kernel | 一批订单的处理规则 | 在设备上执行的并行函数，以及一次对它的启动 |
| grid | 当前完整订单清单 | 一次 kernel 启动产生的全部 block |
| block | 不可拆分的一箱任务 | 一组能够共享片上资源并进行 block 内协作的线程；整体驻留在一个 MP |
| MPC | 物流园分区 | GPU 中较高层的处理器集群，组织多个 MPX |
| MPX | 相邻车间组 | MPC 内的执行引擎层，继续组织多个 MP，并共享部分资源 |
| MP | 真正干活的车间 | block 驻留、warp 被组织和执行的主要 SIMT 处理器 |
| warp | 同步行动的作业小队 | MP 组织线程进行 SIMT 执行的分组 |
| thread | 单个工人 | 执行 kernel 中一个逻辑实例的线程 |

这个类比只解释“谁包含谁”和“工作大致怎样流动”。它不描述缓存一致性、芯片互连拓扑，也不代表精确的调度过程；离开这些边界，物流园就不再是硬件说明书。

## MUSA 的三层硬件组织

### MPC：MUSA Processor Cluster

MPC 是 GPU 内较高层的处理器集群。公开的 MP_10 示例包含 4 个 MPC，并描述了由 MPC 共享的 L2；公开的 MP_21 示例则包含 8 个 MPC。这些数量和缓存归属都是架构相关信息，不是所有 MUSA GPU 的共同常数。

### MPX：MUSA Processor eXecution engine

MPX 位于 MPC 与 MP 之间。以 MP_10 为例，每个 MPC 有 2 个 MPX，每个 MPX 再包含 2 个 MP；同一 MPX 中还共享部分 L1 数据缓存和指令缓存。这里的数量与共享范围同样只适用于对应的公开架构说明。

### MP：MUSA Processor

MP 是主要的 SIMT 处理器，也是一个完整 block 的驻留位置。一个 MP 能同时驻留多少 block、warp 和 thread，会受到寄存器、本地存储容量、最大驻留 block 数和最大驻留 thread 数等资源上限共同约束。

MP 内含浮点（FP）、整数（INT）和特殊函数等执行资源；MP_21 的公开示例还列出了 TCE。具体的执行单元数量、缓存与存储容量都随架构变化，调优时应查询目标设备与对应版本的文档。

下面两组计算只是公开架构示例，不是通用公式：

```text
MP_10：4 MPC × 2 MPX/MPC × 2 MP/MPX = 16 MP
MP_21：8 MPC × 2 MPX/MPC × 2 MP/MPX = 32 MP
```

## 和 CUDA 怎样对照

| MUSA 概念 | 用来理解的 CUDA 概念 | 类比能帮助什么 | 为什么不能画等号 |
|---|---|---|---|
| MPC | GPC | 都可帮助理解 GPU 内较高层的处理资源分组 | 两者的资源边界、工作分发方式和公开层次定义不同 |
| MPX | SM sub-partition | 都能帮助形成“执行资源还会分组”的直觉 | MPX 位于多个 MP 之上，而 SM sub-partition 位于一个 SM 内部，层级方向不同 |
| MP | SM | block 在此驻留，warp 在此被调度和执行 | 执行单元、warp 宽度、缓存组织和调度器设计可能不同 |
| warp | warp | 都是组织线程进行 SIMT 执行的分组 | 宽度以及同步、投票、shuffle 等原语的语义取决于目标平台 |

**编程角色可以类比，物理结构不能逐层翻译。**

## 一个 kernel 的旅行

1. CPU 提交一次 kernel 启动，指定 grid 和 block 的形状。
2. GPU 前端接收这项工作，并按设备实现完成排队与分发。
3. grid 中尚未运行的 block 等待可用执行资源。
4. 调度条件满足后，一个完整 block 被分配到某个 MP；它不会跨 MP 拆分。
5. MP 将 block 中的线程组织成 warp。
6. 调度逻辑从已经就绪的 warp 中选择一个推进执行。
7. warp 的指令进入 FP、INT、访存等相应流水线和执行资源。
8. 某个 warp 等待数据或依赖时，MP 可以推进其他已经就绪的 warp，以隐藏等待时间。

因此，block 是资源分配与线程协作的边界，warp 是 SIMT 执行分组。程序不能依赖 block 的执行先后顺序，不能假设若干 block 必然同时执行，也不能依赖某个 block 被分配到特定 MP；需要跨 block 协作时，应使用编程模型明确提供的机制。

## 这些层级怎样影响性能

### block 大小与尾部 warp

如果 block 的线程数不是目标 warp 大小的整数倍，最后一个 warp 会出现没有对应有效线程的 lane，执行资源可能被浪费。应通过设备属性或对应架构文档查询并确认目标设备的 warp size，再选择 block 大小；不要从 CUDA、另一代 MUSA 架构或旧示例中机械照抄常数。

### 驻留资源与 occupancy

每个 block 使用更多寄存器或片上存储，可能减少一个 MP 能同时驻留的 block 和 warp 数量。反过来，更多已经就绪的 warp 往往能在部分 warp 等待访存或依赖时继续推进，从而隐藏延迟。这里的 occupancy 指实际驻留量相对硬件上限的比例，而不是单独决定速度的分数。

### occupancy 不是最终目标

更高 occupancy 不保证 kernel 更快。分析性能时还要一起看 stall 原因、内存带宽、指令吞吐以及数据复用；有时减少访存、提高复用或降低指令数量，比继续追高 occupancy 更重要。

## 常见误区

- **把软件组织当成固定物理层级。** grid、block、thread 描述程序如何组织并行工作，并不等于芯片中固定的物理层次；运行时会把软件工作映射到可用硬件。
- **把跨平台术语逐层画等号。** MPC ≠ GPC，MPX ≠ SM sub-partition，MP 也不等于 CUDA SM 的全部实现细节；这些名称只能在明确限定下帮助比较编程角色。
- **把教学用语当成模块名。** “前端调度器”或“前端”只是描述接收、排队、分发工作的通用措辞，不能据此发明或断言未公开的内部模块名称。
- **把单一架构参数推广到所有设备。** 某一代架构的 MPC/MPX/MP 数量、缓存归属、容量和执行单元配置不能直接推广到其他架构。
- **把 occupancy 当成最终成绩。** 更高 occupancy 只代表更多潜在并发驻留，不保证更快的 kernel；瓶颈也可能是带宽、依赖、指令吞吐或复用不足。

## 参考资料

- [MUSA Programming Guide：Hardware Architecture（RC 4.3）](https://docs.mthreads.com/musa-sdk/version-4.3.x/programming_guide/Chapter02/)
- [MUSA Programming Guide：GPU Parallel Computing（4.3.x）](https://docs.mthreads.com/musa-sdk/version-4.3.x/programming_guide/Chapter01/)
- [CUDA Programming Guide：Programming Model](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/programming-model.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

## 接下来读什么

- [MUSA 基础概念](concepts.md)：继续理解 SIMT、线程索引、内存层级与同步。
- [CUDA → MUSA 对照与迁移](cuda-vs-musa.md)：查看工具链、API 和迁移差异。
- [MUSA / CUDA 高频混淆点](musa-cuda-pitfalls.md)：复习 warp、同步、访存和多卡场景中的常见陷阱。
