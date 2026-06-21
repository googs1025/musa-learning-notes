# Week 2 · Stream / Graph 示例集 · Design Spec

- **日期**:2026-05-16
- **对应章节**:官方编程指南 Ch5(编程模型 + Stream)
- **状态**:approved,等开干
- **关联 roadmap 行**:Week 2 · 8 示例

---

## 1. 目标

把 Week 2 的 8 个 .mu 示例 + 配套构建文件落到仓库,完成度对齐 Week 1。
**本轮范围(用户选 B)**:**代码 + CMakeLists + Makefile + 简短 README**。
**本轮不做**:`week2/exercises.md`、`docs/articles/02-stream-graph.md`——等用户在 AutoDL 跑通拿到实测数据再补。

---

## 2. 文件清单与每文件设计

每个 .mu 沿用 Week 1 的 **三段式注释**:

```
PART I  ─ 知识点  (4–6 个 § 小节)
PART II ─ 代码实现 (引 musa_common.h,用 MUSA_CHECK / GpuTimer)
PART III ─ Q&A     (2–3 个 ★Q,具体 ms 数字标 "预计 X 量级",注释里写 TODO 等回填)
```

### 2.1 `01_vector_add_runtime.mu`

- **主题**:Runtime API 完整 vectorAdd——"GPU 程序 7 步骨架"第一次完整出现。
- **PART I 知识点**:
  - 7 步骨架(算 bytes / host malloc / device musaMalloc / H2D / kernel / D2H / free)
  - kernel 启动配置:`<<<grid, block>>>`、threadsPerBlock 选 256 的原因
  - 边界保护 `if (i < N)`——为什么不能省
  - 验证正确性:host 端再算一遍对比
- **PART II 代码**:`N = 1<<20`,`vector_add` kernel(签名跟 leetgpu 01 一致),完整 main。
- **PART III Q&A**:
  - Q1:把 `threadsPerBlock` 改成 1024 / 32 会怎样?
  - Q2:省掉 `if (i < N)` 在 N 是 32 倍数时还正确吗?在 N=1023 时呢?

### 2.2 `02_vector_add_pinned.mu`(代替原 Driver API #02)

- **主题**:pinned host memory 让 H2D 拷贝变快、并解锁真异步。
- **PART I 知识点**:
  - 普通 malloc 是 pageable memory,DMA 时驱动得先拷一遍到 staging area
  - `musaMallocHost` / `musaHostAlloc` 直接拿 page-locked memory,驱动 zero-copy DMA
  - pinned 内存的代价(占系统物理页、不能 swap)
  - 是 #05 多流流水线 / `musaMemcpyAsync` 真正异步的**必要条件**
- **PART II 代码**:
  - 同样的 N,跑两遍 H2D:一遍 `malloc` 源,一遍 `musaMallocHost` 源
  - 用 `GpuTimer` 量 `musaMemcpy` 时间,打印加速比
  - 取多次平均(比如 5 次取后 4 次的均值,排除首次冷启动)
- **PART III Q&A**:
  - Q1:加速比通常是几倍?为什么 H2D 比 D2H 更受益?(TODO 实测)
  - Q2:能不能把整个进程都用 pinned?(不能,会拖累 OS 调度)

### 2.3 `03_vector_add_timer.mu`

- **主题**:正确量 kernel 时间——`musaEvent` 而不是 wall clock。
- **PART I 知识点**:
  - 复习 06_async_kernel 的"kernel 是异步的"
  - CPU 计时只能量 launch+sync,GPU 计时器才量真实 kernel 时间
  - 介绍 `musa_common.h::GpuTimer`(就是 `musaEvent` 包装)
  - warm-up:首次 launch 慢,要丢掉
- **PART II 代码**:对同一个 vector_add,
  - 跑法 A:`CpuTimer`(不 sync 一次、有 sync 一次)
  - 跑法 B:`GpuTimer`(包 kernel 那一行)
  - 重复 N=10 次,丢首次,打 min / avg / max
- **PART III Q&A**:
  - Q1:为什么"不 sync 的 CpuTimer" 量出来几乎是 0 ms?
  - Q2:`GpuTimer` 在哪种情况下也会量错?(跨 stream / 没 record)

### 2.4 `04_vector_add_unified.mu`

- **主题**:统一内存(`musaMallocManaged`)——一份指针 host/device 都能用。
- **PART I 知识点**:
  - 统一内存的语义:一个指针,运行时管 page migration
  - 写法对比:省掉 `musaMemcpy` H2D/D2H,代码短一半
  - 性能代价:首次访问会缺页,数据按需迁移
  - `musaMemPrefetchAsync` / `musaMemAdvise` 是优化手段,显式 prefetch 可以避免冷启动
- **PART II 代码**:
  - 一版裸 managed(不 prefetch)
  - 一版加 `musaMemPrefetchAsync` 把数据先搬到 device
  - 用 GpuTimer 量两版的 kernel 时间,看 prefetch 收益
- **PART III Q&A**:
  - Q1:统一内存什么时候真有用?(代码原型 / 大于显存的数据集 / oversubscribe)
  - Q2:训练框架(torch_musa)为什么默认不用统一内存?

### 2.5 `05_multi_stream.mu`

- **主题**:多流流水线——H2D / Kernel / D2H 三阶段重叠。
- **PART I 知识点**:
  - 默认流是同步的(NULL stream),多流才能并发
  - `musaStreamCreate` 创建非默认流
  - 真异步需要 ①pinned host mem ②`musaMemcpyAsync` ③不同 stream
  - 流水线模式:把 N 切 K 份(K=4),3 个 stream 轮转
  - 期望:理想情况下总耗时从 `T_H2D + T_K + T_D2H` 降到 `≈ max(...)`
- **PART II 代码**:
  - `N = 1<<22`(够大,看到重叠效果)
  - 切 4 chunk,4 个 stream,每个 chunk 跑 H2D/Kernel/D2H
  - 用 GpuTimer 量整体时间,对比单流串行版
- **PART III Q&A**:
  - Q1:为什么必须 pinned?(`musaMemcpyAsync` 用 pageable 时退化成同步)
  - Q2:chunk 数 / stream 数怎么选?(经验:stream 数 ≥ 3,chunk 数 ≥ stream 数)

### 2.6 `06_stream_event_dep.mu`

- **主题**:跨流通过 event 同步——形成 DAG 依赖。
- **PART I 知识点**:
  - 同 stream 内任务严格有序;不同 stream 之间默认完全独立
  - 想让 s2 上的 kernel 等 s1 上的 kernel 完成,要用 event 当"红绿灯"
  - 三步:`musaEventCreate` / 在 s1 上 `musaEventRecord` / 在 s2 上 `musaStreamWaitEvent`
  - 注意 `musaStreamWaitEvent` 是 host 立即返回的(只标依赖,不阻塞 host)
- **PART II 代码**:
  - s1:kernel A(写 d_a)
  - 在 s1 上 record event_a_done
  - s2:`musaStreamWaitEvent(s2, event_a_done)` → kernel B(读 d_a 写 d_b)
  - 不加 wait_event 的对照组:可能读到没写完的 d_a(注释里说"实践中你不会真这么写")
- **PART III Q&A**:
  - Q1:为什么不直接 `musaStreamSynchronize(s1)`?(那是 host 阻塞,丢失并发)
  - Q2:event 能跨 device 用吗?(可以,但要特定 flag 创建)

### 2.7 `07_musa_graph.mu`

- **主题**:Stream Capture → Graph,量化反复 launch 的 overhead 节省。
- **PART I 知识点**:
  - launch overhead 是定值(~µs 级,见 Week 1 #06)
  - 当你需要反复 launch 极小 kernel(典型:推理逐 token、迭代求解),launch 成本会主导
  - Graph 解法:把一段 stream 操作录制成 DAG,以后一次 launch 整张图
  - 三步:`musaStreamBeginCapture` / `musaStreamEndCapture(&graph)` / `musaGraphInstantiate(&exec, graph, ...)` / 反复 `musaGraphLaunch(exec, stream)`
  - **a-hamdi/GPU(100 days)有一天专门做了这个对比实验,本例 setup 思路致敬,代码自写**——链接挂在注释里
- **PART II 代码**:
  - 一个极小 kernel(N=1024,只做一次加法,跑得很快)
  - 跑法 A:循环 10000 次直接 `<<<>>>` launch
  - 跑法 B:第一次 capture 录图,之后循环 10000 次 `musaGraphLaunch`
  - 用 GpuTimer 包整段,对比总时间
- **PART III Q&A**:
  - Q1:加速比预计多少?(TODO 实测,理论上 launch 越多收益越大)
  - Q2:Graph 不能动态改吗?(`musaGraphExecUpdate` 可以微调参数,结构改了就要重建)

### 2.8 `08_stream_callback.mu`

- **主题**:GPU 完成时 host 回调——给流水线 join 进 host 逻辑。
- **PART I 知识点**:
  - `musaStreamAddCallback(stream, fn, userData, 0)`
  - 回调函数签名:`void MUSART_CB cb(musaStream_t s, musaError_t err, void* userData)`
  - 回调在 GPU 跑到那个位置时**由 driver 线程**触发,不在 main 上下文
  - **回调里禁止再调 MUSA API**(会死锁),只做轻量 host 工作(打日志 / 通知队列)
  - 用途:多流流水线里,每个 chunk 跑完通知 host 端的下游处理(比如把结果写文件)
- **PART II 代码**:
  - 准备 4 个 chunk,每个 chunk 跑完一个 kernel 后挂一个 callback
  - callback 里 atomic 累加一个全局计数器、打印 "chunk X done at ..."
  - main 等所有 callback 完成后退出
- **PART III Q&A**:
  - Q1:callback 顺序保证吗?(同 stream 内 FIFO;跨 stream 看 driver)
  - Q2:为什么禁止在 callback 里调 musaMalloc 等?(driver 持锁,reenter 死锁)

---

## 3. 构建产物

### 3.1 `code/CMakeLists.txt`(修改)
反注释 `add_subdirectory(week2)`,其余不动。

### 3.2 `code/week2/CMakeLists.txt`(新增)
完全照 Week 1 模板,数组改成 8 个 target 名,foreach 调 `add_musa_executable`(顶层注入)。

### 3.3 `code/week2/Makefile`(新增)
照 `code/week1/Makefile` 抄,BINS 替换成 Week 2 的 8 个,MCC / MUSA_PATH / LDFLAGS 全用变量,允许 `MCC=... MUSA_PATH=... make` 覆盖。

### 3.4 `code/week2/README.md`(新增)
模板沿用 Week 1:

```
# Week 2 · 编程模型 + Stream + Graph
对应官方指南 Ch5。

## 示例表(8 行)
| 文件 | 内容 |
| ... | ... |

## 编译运行
### 方法 A：Makefile
### 方法 B：CMake
### 方法 C：本地编辑 + 远程跑

## 为什么没有 Driver API 示例
3-4 句解释 P2 决策:Runtime API 已覆盖应用层 99% 场景；
Driver API 需要把 kernel 单独编 fatbin / mubin,
不是单 .mu 单文件能完整演示的,留待将来单写一篇笔记。

## 习题与文章
exercises.md 与 docs/articles/02-... 待 AutoDL 跑通后补,
PART III Q&A 里的 ms 数字也是占位,实测后回填。
```

### 3.5 `docs/roadmap.md`(修改)
Week 2 表格 #02 行:`02_vector_add_driver.mu Driver API 重写` → `02_vector_add_pinned.mu pinned memory + 异步 H2D`,状态标 🧪。

### 3.6 顶层 `README.md`(本轮不改)
Week 2 状态行的 ⏳ 留到跑通后再改 ✅。

---

## 4. 不在范围内(out of scope)

- `code/week2/exercises.md`(等实测数据)
- `docs/articles/02-stream-graph.md`(等实测数据 + 跑出图表)
- Driver API 示例(后续单独写)
- 性能调优 / 完整 benchmark 表格

---

## 5. 风险与已知未验证项

| 风险 | 缓解 |
|---|---|
| API 字符串猜错(参数顺序、flag 名) | 已 SSH 上 AutoDL 列了头文件,核心 API 名全部存在;实际 signature 写完后会跑一次远程 `mcc` 编译 sanity check |
| `musaStreamAddCallback` 签名细节(`MUSART_CB` 宏可能叫别的名) | 写完后 grep `musa_runtime_api.h` 拿真名,跑远程编译验证 |
| `musaGraphInstantiate` flag(是否需要 errorNode 参数) | 同上 |
| pinned 内存测时不稳 | 多次取均值 + 丢首次 warm-up |

**所有 ms 数字在 PART III Q&A 都标 "预计 X 量级",注释挂 `// TODO: AutoDL 跑通后回填实测数字"`。**

---

## 6. 完成判据(本轮)

1. 8 个 .mu 全部写完,三段式齐全
2. `code/CMakeLists.txt` 反注释 + `code/week2/CMakeLists.txt` 写好
3. `code/week2/Makefile` 与 `code/week2/README.md` 写好
4. `docs/roadmap.md` #02 行修订
5. **远程 sanity build**:在 AutoDL 跑 `cmake --build` 整个 week2 全部 8 个 target 能编过(不要求跑出正确结果,只验证 API 字符串都对)
6. 把改动 commit(但不 push,等用户决定)
