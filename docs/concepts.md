# MUSA 基础概念

> 跟着 [`code/week1/`](../code/week1/) 的代码读时,这篇文档讲"再多懂一点"的概念。
> 代码注释里只放动手要的最小知识,这里讲完整的心智模型。

---

## 1. SIMT 模型 vs CPU 多线程

| | CPU 多线程 | GPU SIMT |
|---|---|---|
| 线程数量级 | 几个 ~ 几十 | 几千 ~ 几百万 |
| 调度单位 | 单个 thread | **warp**(MUSA = 128 线程,NVIDIA = 32) |
| 指令视角 | 每个线程跑自己的指令流 | warp 内**锁步执行同一条指令** |
| 切换代价 | 高(寄存器保存/恢复)| 极低(寄存器都在 SM 里,切换就是改个指针) |
| 适合任务 | 控制流复杂、分支多 | 数据并行,大量同样的小操作 |

**SIMT** 全称 Single Instruction, Multiple Thread。理解关键:

- 一个 warp 内 128 线程**共享同一条指令的执行**;
- 如果 warp 内的线程走了 `if/else` 不同分支(**warp divergence**),硬件会**串行执行两个分支**(先跑 if 那部分,再跑 else 那部分),其他线程被 mask 掉,等于浪费算力;
- → 写 kernel 时尽量让同一 warp 内的线程走同一条路径。

**为什么 GPU 切换便宜?** 因为它不"切换",而是**等**。当一个 warp 卡在内存访问上,SM 会立刻切到另一个**已经准备好**的 warp 跑指令,寄存器、shared memory 一直在那不动。这就是 GPU 隐藏内存延迟的方式。

→ 所以 SM 上**驻留越多 warp(高 occupancy)越能藏延迟**,这是 Week 4 调优的主线。

---

## 2. 硬件层级:SM → Warp → Block → Thread

```
GPU
 ├── SM 0  (Streaming Multiprocessor)        ← 物理调度单位
 │    ├── Warp scheduler × N                  
 │    ├── Register file (~64 KB)              
 │    ├── Shared memory / L1 (~96 KB)         ← 同 block 共享
 │    └── ALU/FPU/SFU/Tensor 等执行单元        
 ├── SM 1
 ├── ...
 └── SM 47                                     ← MTT S4000 大致这个量级
 
 全局共享:
 ├── L2 Cache (~6 MB)                          
 └── Global Memory (HBM/GDDR, ~48 GB)          
```

软件视角的对应:

| 软件概念 | 硬件对应 |
|---|---|
| `kernel<<<grid, block>>>` | grid 上所有 block 会被分配到各 SM |
| 一个 `block` | 整个 block 调度到**同一个 SM**(不会跨 SM)|
| 一个 `warp`(128 thread) | warp scheduler 调度的最小单位 |
| 一个 `thread` | 占用若干 register,在 ALU 上跑一条 SIMT 指令 |

记住一句话:**block 是调度的边界,warp 是执行的边界。**

---

## 3. 线程层级与全局索引

四个内置变量,在 kernel 内可以直接用:

| 变量 | 含义 | 启动时谁决定 |
|---|---|---|
| `threadIdx` | 线程在 block 内的 (x,y,z) 编号 | 硬件自动分配 |
| `blockIdx`  | block 在 grid 内的 (x,y,z) 编号 | 硬件自动分配 |
| `blockDim`  | 一个 block 的 (x,y,z) 维度 | `<<<grid, block>>>` 第二参数 |
| `gridDim`   | 一个 grid 的 (x,y,z) 维度 | `<<<grid, block>>>` 第一参数 |

### 一维全局索引(99% kernel 从这开始)

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx < N) { ... }    // 边界保护,N 不是 blockDim 整数倍时必备
```

### 二维全局索引

```cpp
int gx = blockIdx.x * blockDim.x + threadIdx.x;
int gy = blockIdx.y * blockDim.y + threadIdx.y;
if (gx < W && gy < H) {
    matrix[gy * W + gx] = ...;   // 行优先存储
}
```

### grid size 的计算(向上取整)

```cpp
int threads = 256;
int blocks  = (N + threads - 1) / threads;
kernel<<<blocks, threads>>>(...);
```

→ 配套代码:[`code/week1/02_thread_index.mu`](../code/week1/02_thread_index.mu)

### 为什么要分 grid + block 两层?

因为 block 内的线程是有"特权"的:

- 共享 **shared memory**(快得多,~30 周期 vs global 的几百周期);
- 可以 `__syncthreads()` 互相等;
- 一定调度到**同一个 SM**。

block 之间没有这些保证 —— 不能直接同步,可以被调到不同 SM 跑。所以:

> **block size 决定协作粒度,grid size 决定总并行度。**

---

## 4. 内存层级

```
速度    │  容量    │  作用域
──────────┼──────────┼──────────────────────
Register │  几十 KB │  单个 thread        ← 最快,~1 周期
Shared   │  ~48 KB  │  block 内共享       ← ~30 周期,要避 bank conflict
L1 Cache │  与 shared 共用                  ↓
L2 Cache │  ~6 MB   │  全 GPU 共享        ← ~200 周期
Constant │  64 KB   │  只读,有专门 cache  ↓
Global   │  ~48 GB  │  全 GPU 可读写      ← ~400-800 周期,要 coalesced
Local    │  在 global 上的"线程私有"        ↓
Host     │  CPU RAM │  通过 musaMemcpy 搬   ← PCIe 慢得多
```

**常见误区**:

- "**Local memory**" 不是真的快内存,它就是给寄存器装不下的局部变量在 global 上开辟的空间,慢。
- "**Constant memory**" 只在所有线程**读同一地址**时通过专用 cache 飞快;每个线程读不同地址就退化成 global。
- 想优化访存性能,核心三件事:
  1. **Coalesced**:同 warp 相邻线程访问相邻地址(week4)
  2. **Shared 替代 Global**:把多次复用的数据搬进 shared(week5 GEMM)
  3. **避免 Bank Conflict**:shared memory 分 32 个 bank,同 warp 内同 bank 不同地址会串行化

---

## 5. 执行模型与同步

### kernel 启动是异步的

```cpp
kernel<<<g, b>>>(...);      // 立即返回,不等 GPU 算完
// CPU 此时可以做别的事
musaDeviceSynchronize();    // 强制等
```

→ 配套代码:[`code/week1/06_async_kernel.mu`](../code/week1/06_async_kernel.mu) 直观看到 launch 时间 ≪ kernel 时间。

### 同步点:把 GPU 的进度拉回 CPU

| 同步方式 | 等什么 | 用法 |
|---|---|---|
| `musaDeviceSynchronize()` | 当前设备所有流上所有任务 | 调试期"暴力同步" |
| `musaStreamSynchronize(s)` | 单个 stream | 多流并发时精确控制 |
| `musaEventSynchronize(e)` | 单个 event 触发 | 计时 + 跨流依赖 |
| `musaMemcpy(D2H)` | **隐式同步**当前流 | 拷数据顺便等 kernel 跑完 |

记住:**没有同步点之前,你不能假设 kernel 已经跑完了**。

### Stream:多任务并发

默认 stream(stream 0)上的任务**串行执行**。要让 H2D / Kernel / D2H 真正重叠,需要多个 stream:

```cpp
musaStream_t s1, s2;
musaStreamCreate(&s1);
musaStreamCreate(&s2);

musaMemcpyAsync(d_in1, h_in1, ..., s1);    // 走 s1
kernel<<<g, b, 0, s2>>>(...);              // 走 s2,可与 s1 并行
```

→ 这是 Week 2 的主菜。

---

## 6. 错误处理:同步 vs 异步

错误分两种,**必须分别抓**:

| 类型 | 例子 | 抓的方式 |
|---|---|---|
| 同步错误 | block 超过 1024 / 参数非法 / 显存不够 | 函数返回值 / `musaGetLastError()` |
| 异步错误 | kernel 内越界 / 写空指针 | 下一个**同步点**才会暴露 |

→ 标准范式:

```cpp
kernel<<<g, b>>>(...);
MUSA_CHECK(musaGetLastError());        // 同步错误
MUSA_CHECK(musaDeviceSynchronize());   // 异步错误
```

- 只 sync 不 GetLastError → launch 失败 kernel 根本没跑,sync 啥也不报
- 只 GetLastError 不 sync → kernel 越界 → 报错点会出现在下次 musaMemcpy

→ 配套代码:[`code/week1/05_error_check.mu`](../code/week1/05_error_check.mu) 主动触发 4 种错误。

### 常见错误码

| 码 | 名 | 含义 |
|---|---|---|
| 0 | `musaSuccess` | 一切正常 |
| 1 | `musaErrorInvalidValue` | 参数非法 |
| 2 | `musaErrorMemoryAllocation` | 显存不够 |
| 9 | `musaErrorInvalidConfiguration` | block/grid 超限 |
| 700 | `musaErrorIllegalAddress` | kernel 越界(异步)|

完整列表见官方编程指南附录。

---

## 7. Host vs Device:两片独立内存

> 这是新手最容易踩的坑,单独拎出来。

```cpp
float *h_ptr = (float*)malloc(N * sizeof(float));   // host RAM
float *d_ptr;
musaMalloc(&d_ptr, N * sizeof(float));               // device VRAM

d_ptr[0] = 1.0f;        // ✗ Segfault(host 解引用 device 地址)
kernel<<<g, b>>>(h_ptr); // ✗ illegal address(kernel 用 host 地址)
```

规则:**host 指针只在 host 用,device 指针只在 device 用,搬数据必须 musaMemcpy。**

命名建议:`h_` 前缀给 host 指针,`d_` 前缀给 device 指针,从一开始就养习惯。

→ 配套代码:[`code/week1/04_memory_basics.mu`](../code/week1/04_memory_basics.mu)

---

## 延伸阅读

- [`cuda-vs-musa.md`](cuda-vs-musa.md) — CUDA / MUSA 命名对照与差异
- [`glossary.md`](glossary.md) — 术语小词典
- [官方编程指南 Ch1–4](https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/programming_guide/) — 权威文档
- [《Programming Massively Parallel Processors》](https://shop.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) — GPU 编程的圣经,值得通读