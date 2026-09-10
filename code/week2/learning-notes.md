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

| 示例 | 必须掌握                                          | 常见误区 |
|---|-----------------------------------------------|---|
| `01_vector_add_runtime.mu` | 算 byte、 host 分配、device 分配、H2D、kernel、D2H、验证、释放 | 跳过小输入验证直接跑大数据 |
| `02_vector_add_pinned.mu` | `musaMallocHost`、DMA、pageable vs pinned       | 以为 pinned 总是越多越好 |
| `03_vector_add_timer.mu` | CPU 计时必须同步，GPU event 更适合 kernel 时间            | 把 launch 入队时间当 kernel 时间 |
| `04_vector_add_unified.mu` | `musaMallocManaged`、prefetch 能力探测             | 以为 UM 一定自动更快 |
| `05_multi_stream.mu` | 多流流水线和 chunk 切分                               | 乱加 device sync 破坏并发 |
| `06_stream_event_dep.mu` | event record/wait 形成 DAG                      | 用全设备同步表达局部依赖 |
| `07_musa_graph.mu` | stream capture、instantiate、launch replay      | 直接套 CUDA graph 性能直觉 |
| `08_stream_callback.mu` | callback 由完成时机触发                              | 以为回调顺序等于提交顺序 |

## 逐示例课文

### 01_vector_add_runtime.mu：Runtime API 的七步骨架

#### 示例目标

这个示例把一个只会写 kernel 的 Week 1 程序补成完整的 host/device 程序：host 准备 `A`、`B`，device 计算 `C=A+B`，再把结果取回并验证。它要建立的是 Runtime API 的固定心智模型，而不是追求复杂优化。

#### 代码结构

`vector_add` 是逐元素 kernel：线程号映射到数组下标，并用 `if (i < N)` 做边界保护。`main` 按七步展开：计算 `bytes`；用 `malloc` 分配并初始化 host buffer；用三次 `musaMalloc` 分配 device buffer；两次同步 H2D；配置 `threadsPerBlock=256` 和向上取整的 `blocksPerGrid` 后启动 kernel；同步 D2H；比较结果并分别释放 device/host 内存。`MUSA_CHECK_KERNEL()` 用来尽早检查 kernel 错误。

#### 核心知识点

七步是：1）算字节数；2）host 分配；3）device 分配；4）H2D；5）kernel；6）D2H；7）释放。向上取整会让最后一个 block 可能多出线程，所以边界判断不能省。这里的 `musaMemcpy` 是同步 API，D2H 返回时通常已经等过前面的 kernel，因此它同时构成了一个容易理解的同步点；这不等于所有 memcpy 都天然同步。

#### 执行流程

host 先填入 `h_A[i]=i`、`h_B[i]=2*i`，把两份输入复制到 device；kernel 让每个有效线程写一个 `d_C[i]`；`MUSA_CHECK_KERNEL()` 检查启动/异步错误；同步 D2H 后 host 重算期望值，最多打印五个 mismatch，最后输出 `OK` 或 `FAILED` 并清理资源。

#### 常见错误与实验

删掉 `if (i < N)`，把 `N` 改成 1023，观察额外线程如何越界；把 `threadsPerBlock` 改成 128、256、512，比较覆盖范围和耗时，不要把某个尺寸的偶然结果当成普遍规律。还可以把同步 D2H 换成 async 版本，配合 stream 和正确的 host buffer 重新思考“入队”和“完成”的区别。

#### 与本周其他示例的关系

这是所有后续例子的基线：`02` 改 host 内存类型，`03` 改计时方法，`04` 改分配/迁移模型，`05` 把同一工作拆成 chunk 和多 stream，`06` 给跨 stream 数据依赖加 event，`07` 把重复操作录成 graph，`08` 在 stream 进度点通知 host。

### 02_vector_add_pinned.mu：pageable 与 pinned host memory

#### 示例目标

这个示例回答“普通 host 内存和 page-locked host 内存的 H2D 路径有什么差异”。它只改变 host buffer 的分配方式，在相同大小的输入上测 pageable 与 pinned 的 H2D 时间和带宽，给 `05` 的真异步拷贝打基础。

#### 代码结构

`bench_h2d` 接收 host 指针、device 目标、字节数和迭代次数，用 `GpuTimer` 包住同步的 `musaMemcpy`，丢弃第一次结果并平均后五次。`main` 先用普通 `malloc` 建立 pageable 组，再用 `musaMallocHost` 建立 pinned 组；两组共用一个 device buffer，最后分别用 `free` 和 `musaFreeHost` 释放。

#### 核心知识点

pageable 内存的物理页可被 OS 管理，驱动通常需要先复制到内部 staging 的 pinned 区，再由 DMA 搬到 GPU；pinned/page-locked 内存可让 DMA 直接访问，减少中转。pinned 不是“越多越快”：它占住物理页，分配/释放也有成本，应在初始化时申请并复用。此文件当前 benchmark 使用的是同步 `musaMemcpy`，因此测的是两种同步 H2D 路径的差异；“pinned 是异步拷贝的前提”要结合 `musaMemcpyAsync` 和 non-default stream，在 `05` 中才完整体现。

#### 执行流程

device 先申请 `16 MB` 目标；pageable 输入初始化后重复 H2D 六次，丢弃冷启动；pinned 输入再重复六次；程序根据平均毫秒数换算带宽并打印 speedup，随后释放全部 buffer。数据和驱动、链路、负载有关，表格中的数值只应当作为本机实验记录。

#### 常见错误与实验

把 pinned 当作任意 host 内存的替代品，或一次性 pin 住整个模型权重，都会造成系统内存压力；小 buffer 也可能被分配成本和测量噪声淹没。可将 `musaMemcpy` 改为 `musaMemcpyAsync`，分别搭配 `malloc`、`musaMallocHost` 和两个 non-default stream，再用同步点确认是否真的异步；同时改变传输大小，观察带宽何时趋于稳定。

#### 与本周其他示例的关系

它继承 `01` 的显式 H2D/D2H 思路，但暂时只聚焦 host 端传输。`03` 解释为什么这里用 GPU event 计时，`05` 依赖 `musaMallocHost` 才能让 H2D/D2H 与 kernel 有机会重叠；`04` 则展示不走显式 H2D 的统一内存方案。

### 03_vector_add_timer.mu：CPU wall-clock 与 GPU event

#### 示例目标

这个示例解决“为什么用 CPU 时钟直接包住 kernel 会得到假数字”。它把同一个 vector add 分成三种测法，区分 launch 入队成本、CPU 等待总耗时和 GPU 时间线上的 kernel 执行时间。

#### 代码结构

为了聚焦计时，程序直接在 device 上 `musaMemset` 初始化 `d_A/d_B`，不测 H2D。`CpuTimer` 的 A 组只包 kernel launch；B 组每次 launch 后调用 `musaDeviceSynchronize()`；C 组使用 `GpuTimer`，内部以同一 stream 上的 start/stop `musaEvent` 计时。B/C 都先做两次 warm-up，再测十次并输出 avg/min/max。

#### 核心知识点

kernel launch 通常是异步入队，CPU 立刻返回，所以无同步的 CPU wall-clock 主要量到 launch 和 driver bookkeeping，不是 kernel。CPU + `musaDeviceSynchronize()` 能覆盖 GPU 完成，但把 host 等待、同步和 CPU 计时开销也算进去。GPU event 是插在 stream 时间线上的 marker，GPU 执行到 marker 才记录时间；start、kernel、stop 必须在同一条相关 stream 上，读取 elapsed 前要等 stop 完成。warm-up 和多次统计用于避开首次初始化及频率变化的影响。

#### 执行流程

程序先跑 A 并同步清空队列；B 先 warm-up，再逐次 CPU 开始、launch、device synchronize、结束；C 再 warm-up，逐次 `g.start()`、launch、`g.stop()`，通过 `elapsed_ms()` 等 event 完成。最后释放 device buffer。一般 B 与 C 接近，A 不具有 kernel 真值含义。

#### 常见错误与实验

不要把无同步的 CPU 时间当 kernel 时间，也不要把跨 stream 的两个 event 当成一段有序区间。可把 `N` 改小，让 launch overhead 更明显；改变 warm-up 次数和迭代次数；把 event 放到错误 stream，观察结果为何不再代表目标 kernel。对跨 stream 总耗时，不能直接套用这个默认 stream 上的 `GpuTimer`。

#### 与本周其他示例的关系

`02` 用同类 GPU event 测 H2D，`04` 用它测统一内存首次访问/迁移效果；`05` 当前特意改用 CPU wall-clock 包住所有 stream 的完整流水线，避免只测到某一条 stream；`06` 的 event 首要用途是依赖，不应和这里的计时 event 混为一谈。

### 04_vector_add_unified.mu：统一内存与按需迁移

#### 示例目标

这个示例展示 `musaMallocManaged` 如何让同一个指针被 host 和 device 使用，同时测量显式 prefetch 对首次 device 访问的影响。它要说明的是易用性和可控性能之间的权衡，而不是证明统一内存必然更快。

#### 代码结构

`run_unified` 为 `A/B/C` 申请 managed memory，host 直接初始化，按设备能力探测 `musaMemPrefetchAsync`；支持时把三块数据预取到 device，随后启动 vector add，并用 `GpuTimer` 测 kernel，再由 host 读取结果验证。`main` 对 no-prefetch 和请求 prefetch 两种模式各跑第一次/第二次 run；每次 run 都会重新申请并初始化内存，不能直接把两次称为冷启动/热启动，最后由 `musaFree` 释放 managed pointer。

README 的实测表沿用“热启动”标签，但由于每次 `run_unified` 都会重新分配、初始化并释放 managed memory，`run 2` 应理解为第二次独立运行，不是真正复用同一批页的热启动。

#### 核心知识点

统一内存把地址语义简化为“一套指针”，但物理页仍可能在 host/device 间迁移：host 初始化后，kernel 首次访问可能触发 page fault 和迁移，host 再读又可能触发回迁。`musaMemPrefetchAsync` 是提前提示位置，减少 kernel 临场 fault；它是可选能力，当前设备可能返回不支持并跳过。代码更短不代表迁移成本消失，性能关键路径通常仍偏好显式 device allocation 加 pinned buffer/async pipeline。

#### 执行流程

每次 `run_unified` 都重新申请三块 managed memory并在 host 初始化；请求 prefetch 时先探测并尝试预取，再运行 kernel、用 event 计时、host 读取 `C` 验证。两种模式各跑两轮后打印 kernel 毫秒数和验证结果；因为每次函数都会重新申请和初始化，所谓第二轮并不是复用同一批已经热在 device 的页。

#### 常见错误与实验

不要把“prefetch 不支持”误判为代码错误，也不要把一次设备上的结果外推到所有 MUSA SDK。可在支持/不支持 prefetch 的设备上对比输出；改变 `N` 和 host/device 交替访问次数，观察 page migration 对延迟和方差的影响；再和 `01` 的显式 H2D/D2H 做同数据规模对照。

#### 与本周其他示例的关系

它是 `01` 七步骨架的简化分配路径，与 `02` 的 pinned host 路线形成对照。`03` 提供它所用的 event 计时方法，`05` 则回到显式 pinned buffer 来换取更可预测的多 stream pipeline；`07` 的 graph 更适合固定操作结构，不能自动消除统一内存迁移。

### 05_multi_stream.mu：按 chunk 组织流水线

#### 示例目标

这个示例探究如何把一次大 vector add 切成多个 chunk，让 H2D、kernel 和 D2H 在不同 stream 中有机会重叠。它的目标是观察调度条件和流水线形状，不是承诺多 stream 一定加速。

#### 代码结构

程序设 `N=1<<24`、`CHUNKS=4`，以 `n=N/CHUNKS` 和 `cbytes=bytes/CHUNKS` 切分；host 的 `h_A/h_B/h_C` 全部由 `musaMallocHost` 申请，device 的 `d_A/d_B/d_C` 各是一整块、并不按 chunk 分配。A 组在默认流用同步 memcpy、一个完整 kernel 和同步 D2H；B 组创建四条 stream，每条依次提交两次 H2D、一个 chunk kernel 和一次 D2H。当前文件的 B 组用 `CpuTimer` 包住“全部入队到所有 stream synchronize 完成”的 CPU wall-clock。

#### 核心知识点

真正异步需要 pinned host memory、`musaMemcpyAsync` 和 non-default stream 三者同时成立。每个 chunk 的 `off=c*n`、`d_*+off`、`h_*+off` 都指向不重叠的数据范围：chunk `c` 只读写 `[c*n,(c+1)*n)`，因此不同 stream 没有数据竞争。单条 stream 内仍保持 H2D→H2D→kernel→D2H 顺序；跨 stream 只有在硬件资源允许时才会并发。并发可能降低总耗时，也可能受 copy engine 数量、kernel 占用、调度、传输粒度或 launch overhead 限制，不能把“用了四条流”写成“必然四倍/必然加速”。

#### 执行流程

A 组完成整块传输、计算和回传后记录串行 wall-clock；B 组创建 stream，主线程为每个 chunk 入队 H2D/H2D/kernel/D2H，再逐条 `musaStreamSynchronize`，计时结束并销毁 stream。最后扫描完整 `h_C` 验证 `A+B`。因为同步发生在计时结束前，B 的时间是整条流水线的端到端 CPU wall-clock，不是某一条 GPU stream 的 kernel 时间。

#### 常见错误与实验

把 `malloc` 换回去、把 async 换成同步 memcpy，或在当前默认/legacy stream 语义下把工作放回默认 stream，都会让“异步”退化；non-blocking/per-thread 配置可能改变其中的跨流隐式同步关系。把 `CHUNKS` 改成 1、2、8、16，观察 chunk 太少无法填满阶段、太多则被小任务和 launch 开销拖慢；用 event 给每个 chunk 标出 H2D/kernel/D2H 时间线，验证是否真的重叠。不要只看一次 A/B 数字就下结论，至少重复并记录设备、SDK 和输入规模。

#### 与本周其他示例的关系

`02` 提供 pinned host memory 的背景，`01` 提供每个 chunk 内部仍然遵循的拷贝—kernel—拷贝骨架。`06` 说明如果 chunk 之间出现真实读写依赖，必须用 event；`08` 展示工作完成后如何通知 host；`03` 则解释为什么本例不能用单个默认 stream 的 GPU timer 代替当前 wall-clock。

### 06_stream_event_dep.mu：用 event 表达跨流依赖

#### 示例目标

多 stream 让不同任务可以无序推进，但 `square_to_b` 必须等 `fill_A` 把 `d_a` 填好。这个示例展示如何只同步必要的生产者—消费者边，而不让 host 用全设备同步把整个程序串行化。

#### 代码结构

`fill_A` 在 `s1` 写 `d_a[i]=i+1`，`square_to_b` 在 `s2` 读 `d_a` 写平方。`main` 创建 `s1/s2` 和 `a_done` event，在 `s1` 记录 event；随后在 `s2` 调用 `musaStreamWaitEvent(s2,a_done,0)`，再提交消费者 kernel 和 D2H。D2H 完成后同步 `s2`，host 验证平方结果并销毁 event、streams 和 buffers。

#### 核心知识点

同一 stream 通常 FIFO，跨 stream 默认没有顺序保证。`musaEventRecord` 是把 marker 插入 `s1` 的当前位置，GPU 执行到那里才把 event 标记完成；`musaStreamWaitEvent` 是向 `s2` 的队列加入依赖，调用本身不阻塞 host，只保证 event 之后的 `s2` 工作等待它。这样形成一个小型 DAG：`fill_A → a_done → square_to_b → D2H`。`musaStreamSynchronize` 则是 host 阻塞等待整条 stream，作用范围更大。

#### 执行流程

两条 stream 创建后，生产者 kernel 入队并记录 `a_done`；消费者 stream 先入队 wait，再入队平方 kernel 和 async D2H；host 最后同步 `s2`，读取结果并按相对误差检查。由于 `s2` 上的 D2H 位于消费者之后，它也自然等待消费者完成，但这不会替代生产者到消费者之间的 event 依赖。

#### 常见错误与实验

删掉 `musaStreamWaitEvent` 可能仍因 kernel 太快而“碰巧正确”，这不是依赖成立的证据。可让 `fill_A` 加长空转循环、增大数据量后重复实验，观察潜在 race；再加入第三条无关 stream，确认 event wait 不会阻塞 host 或无故阻塞不相关工作。不要用 `musaDeviceSynchronize()` 代替所有局部依赖。

#### 与本周其他示例的关系

它是 `05` 流水线“各 chunk 独立、无需跨流等待”的补充：一旦有共享 buffer 或阶段依赖，就要把边画成 event DAG。`03` 也用 event，但目标是测时；`07` 会把固定的操作依赖捕获成 graph；`08` 则在一条 stream 到达完成点后通知 host。

### 07_musa_graph.mu：capture、instantiate 与 replay

#### 示例目标

这个示例研究大量小 kernel launch 时，host/driver 的 launch overhead 是否成为主要成本。它比较直接重复提交五个 `add_one` kernel 与把这五个操作 capture 成 graph 后重复 `musaGraphLaunch` 的差异。

#### 代码结构

`N=1024`、`ITERS=5000`、`OPS_PER_STEP=5`，只有一条显式 stream。A 组清零后直接循环提交 `5*ITERS` 个 kernel；B 组再次清零，调用 `musaStreamBeginCapture`，提交五个 kernel，`musaStreamEndCapture` 得到 `graph`，`musaGraphInstantiate` 得到可执行 `exec`，再循环 `ITERS` 次 launch。两组都用同一条 stream 的 `GpuTimer` 计时，最后 D2H 读取 `x[0]` 验证。

#### 核心知识点

capture 阶段记录的是操作和依赖，kernel 不会在录制时真正执行；end capture 后还要 instantiate，之后的 `musaGraphLaunch` 才会真正执行整张图。Graph 把固定拓扑的多个操作压缩成可重复提交的执行对象，可能降低 launch bookkeeping，但 graph launch 本身也有成本，收益取决于 SDK、硬件、图的复杂度和 kernel 粒度。结构或 launch 配置改变通常需要重新构建/实例化，不能把 CUDA 上的性能直觉直接套到 MUSA；本例已有实测显示 graph 可能反而更慢。

#### 执行流程

A 清零、计时并直接提交 25,000 个 kernel；同步后 B 再清零并同步，capture 五个 `add_one`，结束 capture、instantiate，再计时 5,000 次 graph replay；同步后销毁 exec/graph。最终值为 `ITERS*OPS_PER_STEP=25000`，说明 B 的计算来自 replay，而不是 capture。

#### 常见错误与实验

把 capture 当成一次真实计算会误判结果；可在 capture 后立即读取 `d_x` 验证它仍未被五个 kernel 修改。改变 `OPS_PER_STEP`、`N` 和 `ITERS`，分别观察图的固定成本何时摊薄；每次比较都要重新清零、warm-up/同步并记录 MUSA SDK，因为“Graph 一定更快”是不成立的结论。

#### 与本周其他示例的关系

`06` 的 event DAG 是理解 graph 节点依赖的基础，`03` 的同 stream GPU timer 是本例的测量工具。`05` 关注跨 stream 的流水线吞吐，而本例关注固定单 stream DAG 的重复提交；`08` 的 `musaStreamAddCallback` 还涉及 host 回调，不能未经验证地假设它可被 capture。

### 08_stream_callback.mu：GPU 完成点通知 host

#### 示例目标

这个示例解决 host 如何得知某条 stream 已经走到指定完成点，并把 chunk 编号/状态交给 host 逻辑。它演示的是轻量通知机制，不是让 callback 线程替代主线程处理业务。

#### 代码结构

`fill_with` 是向 device buffer 填入 chunk 编号的 kernel。程序创建四个独立 device buffer、四条 stream 和四个 `CbCtx`；每个 stream 入队一个 kernel 后调用 `musaStreamAddCallback(stream,on_chunk_done,&ctx[i],0)`。callback 读取 `userData`，用 `std::atomic<int>` 增加完成计数并打印 chunk、status 和总数；main 随后逐条同步 stream，源码只检查计数是否为 4，并没有把 `status` 纳入成功判定，最后销毁资源。

#### 核心知识点

callback 在所属 stream 到达它在队列中的位置后，由运行时在 host 侧异步执行，不应假定是在 main 线程或某个指定线程上运行；同一 stream 内保持顺序，不同 stream 之间完成顺序不保证等于提交顺序。`userData` 用来传递上下文，`flags` 当前填 0。callback 应只做原子计数、设置通知或投递轻量消息；不要在其中调用 `musaMalloc`、`musaMemcpy` 或再次 launch，也不要做重 CPU 工作或让异常穿过 C 回调边界。回调表示“到达完成点”，并不自动保证跨 stream 的其他依赖已经建立。

#### 执行流程

main 先为四个 chunk 分配 buffer 和 stream，再把各自的填充 kernel 与 callback 按 FIFO 入队；随后同步四条 stream，等待 kernel 和对应 callback 都完成，打印 `counter=4`。跨 stream 的 callback 可能按任意顺序打印，但 atomic 计数的最终值应为四；这里的 `counter=4` 只证明四个 callback 都被调用，不证明四个 callback 的 `status` 都成功，因为源码当前没有把 `status` 纳入成功判定。本例没有 D2H，因为验证重点是通知时机和 callback 状态，而不是数据回传。

#### 常见错误与实验

不要依赖跨 stream callback 的打印顺序来组织文件或下游请求；可重复运行观察不同完成顺序，并把 `CHUNKS` 改变。可在 callback 中只投递 `chunk_id`，让 main/worker 按编号消费；不要把 `musaMemcpy` 塞进 callback 测试“能否顺便做事”，这可能死锁或返回错误。还可对比 `musaLaunchHostFunc` 的简化签名及 graph capture 兼容性，但应以当前 MUSA SDK 文档为准。

#### 与本周其他示例的关系

它承接 `05` 的多 stream 和 `06` 的 stream 完成语义：`05` 负责安排独立 chunk，`08` 负责在某个 chunk 到达完成点时发信号；`06` 仍负责真正的数据依赖。`03` 的 event 是 GPU 内部计时/标记，callback 是 GPU 进度到达后的 host 通知；`07` 的 graph 则关注固定 DAG 的 replay，三者用途不同。

## 代码阅读抓手

本周所有示例都围绕一个问题：任务是否真的重叠。阅读时重点看：

- host memory 是 pageable 还是 pinned。
- memcpy 是同步版本还是 async 版本。
- kernel 和 memcpy 是否在同一个 stream。
- 是否有 event 或 device sync 把本来能并发的工作串行化。

## 高频混淆点

- **异步入队 != 已经完成**: `kernel<<<...>>>` 和 `musaMemcpyAsync` 通常只是把任务提交到队列。对同步的 `musaMemcpy`，函数返回时相应拷贝已完成，D2H 返回后 host 才能安全读取结果；对 `musaMemcpyAsync`，host 必须等对应 stream/event（或更大范围的 device）同步后，才能安全读取其 host 目标 buffer。
- **默认 stream 的串行化取决于语义**: 在当前默认/legacy stream 语义下，把工作都放在默认 stream 可能通过隐式同步限制跨流并发；但 non-blocking stream、per-thread default stream 等配置会改变跨流隐式同步关系。要观察重叠，需确认实际 stream 配置和依赖，而不能只看“是否用了默认 stream”。
- **pinned memory 不是免费午餐**: `musaMallocHost` 有利于 DMA 和异步拷贝, 但 pinned 内存占多了会影响系统内存管理。小数据也可能看不出收益。
- **Event 不是普通 CPU 时间戳**: `musaEventRecord` 记录的是 GPU stream 时间线上的点。计 kernel 时间时要把 start/end event 放到正确 stream。
- **`musaDeviceSynchronize()` 会一刀切**: 调试期好用, 但多 stream 代码里乱加它会把 H2D/kernel/D2H 的重叠全部冲掉。

## 执行模型与硬件层次：SM / grid / block / warp

这几个概念经常一起出现, 但不完全属于同一个层面：

- **grid**: 一次 kernel 启动产生的全部线程块集合, 是整个任务的执行范围。
- **block**: grid 中的一组线程。block 内的线程可以使用 shared memory, 并通过 `__syncthreads()` 协作。
- **warp**: GPU 实际调度线程的基本小组。一个 block 会被拆成多个 warp, warp 内线程通常以相同的指令流执行。
- **SM (Streaming Multiprocessor)**: GPU 内部真正承载和执行 block 的硬件单元, 可以理解为一个“线程块执行场所”。它包含计算核心、寄存器、shared memory 和调度资源。

因此, **grid / block / warp 是编程模型或执行模型中的层次, SM 是硬件结构**。它们不是完全同一层面的概念:

```text
一次 kernel 启动
└── grid
    ├── block 0 ──┬── warp 0
    │             ├── warp 1
    │             └── ...
    ├── block 1 ──┬── warp 0
    │             └── ...
    └── block 2 ...

GPU 硬件调度：
    block 0 ──> SM 0
    block 1 ──> SM 1
    block 2 ──> SM 0   # SM 0 有足够资源时可同时驻留多个 block
```

### 一个简单例子

```cpp
__global__ void vector_add(const float* a, const float* b,
                           float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vector_add<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
```

假设 `n = 1000`, `threadsPerBlock = 256`:

- `blocksPerGrid = 4`, 所以这次启动有 4 个 block, 这 4 个 block 组成一个 grid。
- 每个 block 有 256 个线程, GPU 会进一步把它拆成若干个 warp。
- GPU 调度器再把这些 block 分配到不同 SM 上执行。
- 程序员指定的是 grid 和 block; warp 的具体划分以及 block 何时放到哪个 SM, 通常由硬件和运行时决定。

可以用下面这句话记忆:

```text
grid = 整个 kernel 任务
block = 任务分组
warp = block 内的硬件调度小组
SM = 承载并执行 block 的 GPU 硬件单元
```

### 和 threadsPerBlock 的关系

`threadsPerBlock` 越大, 一个 block 通常需要的寄存器和 shared memory 越多, 一个 SM 能同时驻留的 block 数量可能变少; 但这不代表线程数越大一定越慢。实际最佳值要结合寄存器使用量、shared memory 使用量、内存访问方式和具体 GPU 架构测试。

CUDA 中 warp 通常包含 32 个线程; MUSA 的线程组织和 warp 宽度应以具体 MUSA 架构及 SDK 文档为准, 不要直接把 CUDA 的所有 warp 结论照搬过去。

### Stream 和 grid / block 的关系

`stream` 和 `grid` 也不是同一个层面的概念:

- **grid** 描述“一次 kernel 启动要处理什么任务”, 由一组 block 组成。
- **stream** 描述“任务以什么顺序提交和执行”, 是 GPU 工作队列或时间线。
- 一次 kernel launch 会产生一个 grid, 并被提交到某一个 stream。
- 一个 stream 可以依次提交多个 grid, 也可以提交内存拷贝、event 等操作。
- 同一个 stream 中的操作通常保持提交顺序; 不同 stream 中的操作在资源允许时可能并发。
- stream 不负责把 grid 拆成 block, 也不决定 block 最终运行在哪个 SM 上; 这些由 kernel 的执行配置和 GPU 调度器共同决定。

可以把完整关系记成:

```text
host 代码
  │
  ├── stream 0：H2D copy → kernel launch(grid 0) → D2H copy
  │                         └── grid 0
  │                             └── block → warp → thread
  │
  └── stream 1：H2D copy → kernel launch(grid 1)

GPU 硬件：grid 中的 block 被调度到 SM 上执行
```

例如下面的代码把一次向量加法提交到 `stream`:

```cpp
dim3 block(256);
dim3 grid((N + block.x - 1) / block.x);

musaMemcpyAsync(d_a, h_a, bytes, musaMemcpyHostToDevice, stream);
vector_add<<<grid, block, 0, stream>>>(d_a, d_b, d_c, N);
musaMemcpyAsync(h_c, d_c, bytes, musaMemcpyDeviceToHost, stream);
```

这里有三个不同的概念:

1. `grid` 表示这次 `vector_add` kernel 的全部 block。
2. `block(256)` 表示每个 block 有 256 个线程。
3. `stream` 表示 H2D 拷贝、kernel 和 D2H 拷贝被提交到同一条有序队列中。

因此, 可以这样记忆:

```text
grid / block / warp：描述一次 kernel 如何组织线程
stream：描述多个 GPU 操作如何排队、同步和可能并发
SM：硬件执行 block 的地方
```

## CUDA_Freshman 对照

- `3_sum_arrays`: vector add 主流程。
- `4_sum_arrays_timer`: 计时方法。
- `15_pine_memory`: pinned memory。
- `17_UVA`: 统一虚拟地址/统一内存相关概念。
- `30_stream`、`34_stream_dependence`、`37_asyncAPI`、`38_stream_call_back`: stream、event、async、callback。

完整映射见 [`../../docs/cuda-example-map.md`](../../docs/cuda-example-map.md)。
